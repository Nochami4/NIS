from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from meteostat import Point, config, daily, hourly, stations
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.constants import (
    DEFAULT_START_DATE,
    DEFAULT_MIN_HOURLY_OBSERVATIONS,
    FIGURES_DIR,
    FIXED_DATA_END_DATE,
    MAX_QUALITY_MIN_YEAR_SHARE,
    MODEL_DATASET_FILENAME,
    NEURAL_BATCH_SIZE,
    NEURAL_EPOCHS,
    NEURAL_PATIENCE,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DAILY_FILENAME,
    RAW_DATA_DIR,
    RAW_HOURLY_FILENAME,
    RANDOM_STATE,
    SEASONAL_PERIOD_DAYS,
    SEQUENCE_WINDOW_DAYS,
    TABLES_DIR,
    TEST_DAYS,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_LENGTH_YEARS,
    VALIDATION_DAYS,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
    VOLGOGRAD_ELEVATION,
    VOLGOGRAD_LATITUDE,
    VOLGOGRAD_LONGITUDE,
    VOLGOGRAD_NAME,
)

SPARSE_OPTIONAL_COLUMNS = {"wpgt_max", "tsun_sum", "rhum_mean", "dwpt_mean"}
CORE_REQUIRED_COLUMNS = {
    "target_tavg",
    "tmin",
    "tmax",
    "temp_range",
    "prcp_sum",
    "snow",
    "pres_mean",
    "wspd_mean",
}


def ensure_project_directories() -> None:
    """Создаёт рабочие папки проекта, если они ещё не существуют."""

    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_meteostat_runtime() -> None:
    """Настраивает Meteostat для локального проекта."""

    ensure_project_directories()

    meteostat_root = RAW_DATA_DIR / "meteostat_cache"
    meteostat_cache = meteostat_root / "cache"
    stations_db_file = meteostat_root / "stations.db"

    meteostat_cache.mkdir(parents=True, exist_ok=True)

    config.cache_directory = str(meteostat_cache)
    config.stations_db_file = str(stations_db_file)
    config.block_large_requests = False


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Вычисляет расстояние между двумя точками на Земле в километрах."""

    radius_km = 6371.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return radius_km * c


def _strip_timezone(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Удаляет временную зону из индекса, если она присутствует."""

    if index.tz is None:
        return index
    return index.tz_convert(None)


def normalize_meteostat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Преобразует индекс Meteostat в обычный столбец даты."""

    normalized = frame.copy()

    normalized.index = _strip_timezone(
        pd.DatetimeIndex(pd.to_datetime(normalized.index))
    )

    normalized.index.name = "time"

    normalized = normalized.reset_index()

    normalized["time"] = pd.to_datetime(normalized["time"])
    normalized["date"] = normalized["time"].dt.normalize()

    return normalized


def get_nearby_station_candidates(
    latitude: float = VOLGOGRAD_LATITUDE,
    longitude: float = VOLGOGRAD_LONGITUDE,
    elevation: float = VOLGOGRAD_ELEVATION,
    limit: int = 8,
) -> pd.DataFrame:
    """Возвращает ближайшие к Волгограду станции Meteostat."""

    point = Point(latitude, longitude, int(elevation) if elevation is not None else None)
    station_candidates = stations.nearby(point, limit=limit)
    if station_candidates is None or station_candidates.empty:
        raise RuntimeError("Meteostat не вернул ближайшие станции для выбранных координат.")

    station_candidates = station_candidates.reset_index().rename(columns={"id": "station_id"})
    station_candidates["distance_km"] = station_candidates.apply(
        lambda row: haversine_distance_km(
            latitude,
            longitude,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )
    if "distance" in station_candidates.columns:
        station_candidates["distance_km"] = (
            station_candidates["distance"].astype(float).div(1000).round(2)
        )

    return station_candidates.sort_values(["distance_km", "name"]).reset_index(drop=True)


def fetch_meteostat_data(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    latitude: float = VOLGOGRAD_LATITUDE,
    longitude: float = VOLGOGRAD_LONGITUDE,
    elevation: float = VOLGOGRAD_ELEVATION,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает данные Meteostat и при необходимости переходит на Point fallback."""

    ensure_project_directories()
    configure_meteostat_runtime()

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    hourly_end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    candidates = get_nearby_station_candidates(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
    )

    errors: list[str] = []

    for row in candidates.head(5).itertuples(index=False):
        try:
            station_id = getattr(row, "station_id")
            hourly_data = hourly(station_id, start_ts, hourly_end_ts).fetch()
            daily_data = daily(station_id, start_ts, end_ts).fetch()

            if hourly_data is None or daily_data is None:
                errors.append(f"Станция {station_id}: Meteostat вернул пустой объект.")
                continue

            if hourly_data.empty and daily_data.empty:
                errors.append(f"Станция {station_id}: пустой ответ.")
                continue

            selection = {
                "source_type": "station",
                "station_id": station_id,
                "station_name": getattr(row, "name"),
                "country": getattr(row, "country"),
                "region": getattr(row, "region"),
                "distance_km": float(getattr(row, "distance_km")),
                "latitude": float(getattr(row, "latitude")),
                "longitude": float(getattr(row, "longitude")),
                "elevation": float(getattr(row, "elevation")) if pd.notna(getattr(row, "elevation")) else np.nan,
            }
            return selection, candidates, daily_data, hourly_data
        except Exception as exc:  # pragma: no cover - сеть и внешние данные
            errors.append(f"Станция {getattr(row, 'station_id')}: {exc}")

    try:
        point = Point(latitude, longitude, int(elevation) if elevation is not None else None)
        hourly_data = hourly(point, start_ts, hourly_end_ts).fetch()
        daily_data = daily(point, start_ts, end_ts).fetch()

        if hourly_data is None or daily_data is None:
            raise RuntimeError("Point fallback вернул пустой объект.")

        if hourly_data.empty and daily_data.empty:
            raise RuntimeError("Point fallback вернул пустые данные.")

        selection = {
            "source_type": "point",
            "station_id": "POINT",
            "station_name": VOLGOGRAD_NAME,
            "country": "RU",
            "region": "Волгоградская область",
            "distance_km": 0.0,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
        }
        return selection, candidates, daily_data, hourly_data
    except Exception as exc:  # pragma: no cover - сеть и внешние данные
        joined_errors = "; ".join(errors) if errors else "Подходящие станции не найдены."
        raise RuntimeError(
            "Не удалось загрузить данные Meteostat ни по станции, ни по координатам. "
            f"Подробности: {joined_errors}; Point fallback: {exc}"
        ) from exc


def save_raw_snapshots(hourly: pd.DataFrame, daily: pd.DataFrame) -> tuple[Path, Path]:
    """Сохраняет сырые выгрузки Meteostat в папку data/raw."""

    ensure_project_directories()

    hourly_path = RAW_DATA_DIR / RAW_HOURLY_FILENAME
    daily_path = RAW_DATA_DIR / RAW_DAILY_FILENAME

    normalize_meteostat_frame(hourly).to_csv(hourly_path, index=False)
    normalize_meteostat_frame(daily).to_csv(daily_path, index=False)
    return hourly_path, daily_path


def load_raw_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает ранее сохранённые сырые данные из data/raw."""

    hourly_path = RAW_DATA_DIR / RAW_HOURLY_FILENAME
    daily_path = RAW_DATA_DIR / RAW_DAILY_FILENAME

    if not hourly_path.exists() or not daily_path.exists():
        raise FileNotFoundError("Локальные raw-снимки Meteostat не найдены.")

    hourly_frame = pd.read_csv(hourly_path, parse_dates=["time", "date"]).set_index("time")
    daily_frame = pd.read_csv(daily_path, parse_dates=["time", "date"]).set_index("time")
    return daily_frame, hourly_frame


def subset_raw_snapshots(
    daily_frame: pd.DataFrame,
    hourly_frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ограничивает raw-данные фиксированным временным интервалом."""

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    hourly_end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    daily_subset = daily_frame.loc[(daily_frame.index >= start_ts) & (daily_frame.index <= end_ts)].copy()
    hourly_subset = hourly_frame.loc[(hourly_frame.index >= start_ts) & (hourly_frame.index <= hourly_end_ts)].copy()
    return daily_subset, hourly_subset


def aggregate_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует почасовые данные до уровня суток."""

    hourly_norm = normalize_meteostat_frame(hourly)

    snow_column = "snow" if "snow" in hourly_norm.columns else "snwd" if "snwd" in hourly_norm.columns else None

    aggregation_plan: dict[str, tuple[str, str]] = {
        "target_tavg": ("temp", "mean"),
        "tmin": ("temp", "min"),
        "tmax": ("temp", "max"),
        "prcp_sum": ("prcp", "sum"),
        "pres_mean": ("pres", "mean"),
        "wspd_mean": ("wspd", "mean"),
        "wpgt_max": ("wpgt", "max"),
        "rhum_mean": ("rhum", "mean"),
        "dwpt_mean": ("dwpt", "mean"),
        "tsun_sum": ("tsun", "sum"),
    }
    if snow_column is not None:
        aggregation_plan["snow"] = (snow_column, "max")

    named_aggregations = {
        output_name: pd.NamedAgg(column=source_column, aggfunc=agg_func)
        for output_name, (source_column, agg_func) in aggregation_plan.items()
        if source_column in hourly_norm.columns
    }

    if "temp" in hourly_norm.columns:
        named_aggregations["temp_observations"] = pd.NamedAgg(column="temp", aggfunc="count")

    if not named_aggregations:
        raise RuntimeError("В почасовых данных отсутствуют нужные метеорологические столбцы.")

    aggregated = (
        hourly_norm.groupby("date")
        .agg(**named_aggregations)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if {"tmin", "tmax"}.issubset(aggregated.columns):
        aggregated["temp_range"] = aggregated["tmax"] - aggregated["tmin"]

    return aggregated


def prepare_daily_fallback(daily: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает суточные данные как запасной источник признаков."""

    daily_norm = normalize_meteostat_frame(daily)

    rename_map = {
        "temp": "daily_tavg",
        "tavg": "daily_tavg",
        "tmin": "daily_tmin",
        "tmax": "daily_tmax",
        "prcp": "daily_prcp",
        "snwd": "daily_snow",
        "snow": "daily_snow",
        "pres": "daily_pres",
        "wspd": "daily_wspd",
        "wpgt": "daily_wpgt",
        "rhum": "daily_rhum",
        "tsun": "daily_tsun",
    }

    columns_to_keep = ["date"] + [column for column in rename_map if column in daily_norm.columns]
    fallback = daily_norm[columns_to_keep].rename(columns=rename_map)
    return fallback.sort_values("date").reset_index(drop=True)


def combine_hourly_and_daily_features(
    hourly_daily: pd.DataFrame,
    daily_fallback: pd.DataFrame,
    min_hourly_observations: int = DEFAULT_MIN_HOURLY_OBSERVATIONS,
) -> pd.DataFrame:
    """Объединяет агрегированные hourly-признаки и fallback из Daily."""

    dataset = hourly_daily.merge(daily_fallback, on="date", how="outer").sort_values("date").reset_index(drop=True)

    if "temp_observations" in dataset.columns:
        insufficient_obs = dataset["temp_observations"] < min_hourly_observations
        for column in ["target_tavg", "tmin", "tmax"]:
            if column in dataset.columns:
                dataset.loc[insufficient_obs, column] = np.nan

    fallback_pairs = {
        "target_tavg": "daily_tavg",
        "tmin": "daily_tmin",
        "tmax": "daily_tmax",
        "prcp_sum": "daily_prcp",
        "snow": "daily_snow",
        "pres_mean": "daily_pres",
        "wspd_mean": "daily_wspd",
        "wpgt_max": "daily_wpgt",
        "rhum_mean": "daily_rhum",
        "tsun_sum": "daily_tsun",
    }

    for main_column, fallback_column in fallback_pairs.items():
        if main_column in dataset.columns and fallback_column in dataset.columns:
            dataset[main_column] = dataset[main_column].combine_first(dataset[fallback_column])
        elif main_column not in dataset.columns and fallback_column in dataset.columns:
            dataset[main_column] = dataset[fallback_column]

    if {"tmin", "tmax"}.issubset(dataset.columns):
        dataset["temp_range"] = dataset["tmax"] - dataset["tmin"]

    daily_columns = [column for column in dataset.columns if column.startswith("daily_")]
    return dataset.drop(columns=daily_columns, errors="ignore")


def season_from_month(month: int) -> str:
    """Возвращает название сезона по номеру месяца."""

    if month in [12, 1, 2]:
        return "зима"
    if month in [3, 4, 5]:
        return "весна"
    if month in [6, 7, 8]:
        return "лето"
    return "осень"


def season_to_code(season: str) -> int:
    """Преобразует текстовый сезон в числовой код."""

    mapping = {"зима": 0, "весна": 1, "лето": 2, "осень": 3}
    return mapping[season]


def add_calendar_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные и циклические признаки."""

    enriched = dataset.copy()
    enriched["month"] = enriched["date"].dt.month
    enriched["dayofyear"] = enriched["date"].dt.dayofyear
    enriched["dayofweek"] = enriched["date"].dt.dayofweek
    enriched["season"] = enriched["month"].apply(season_from_month)
    enriched["season_code"] = enriched["season"].apply(season_to_code)
    enriched["doy_sin"] = np.sin(2 * np.pi * enriched["dayofyear"] / 365.25)
    enriched["doy_cos"] = np.cos(2 * np.pi * enriched["dayofyear"] / 365.25)
    return enriched


def add_target_history_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Добавляет лаговые и скользящие признаки по температурным аномалиям."""

    enriched = dataset.copy()
    history = enriched["target_anomaly"].shift(1)

    for lag in [1, 2, 3, 7, 14, 30]:
        enriched[f"lag_{lag}"] = enriched["target_anomaly"].shift(lag)

    enriched["rolling_mean_3"] = history.rolling(window=3).mean()
    enriched["rolling_mean_7"] = history.rolling(window=7).mean()
    enriched["rolling_mean_14"] = history.rolling(window=14).mean()
    enriched["rolling_std_7"] = history.rolling(window=7).std()
    enriched["rolling_std_14"] = history.rolling(window=14).std()

    return enriched


def apply_missing_value_strategy(dataset: pd.DataFrame) -> pd.DataFrame:
    """Аккуратно обрабатывает пропуски и подготавливает базовые признаки."""

    cleaned = dataset.copy().sort_values("date").reset_index(drop=True)

    for column in ["prcp_sum", "snow", "tsun_sum"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0.0)

    one_sided_fill_columns = [
        "tmin", "tmax", "temp_range", "prcp_sum", "snow",
        "pres_mean", "wspd_mean", "wpgt_max", "rhum_mean",
        "dwpt_mean", "tsun_sum",
    ]

    for column in one_sided_fill_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].ffill(limit=7)

    cleaned = cleaned.dropna(subset=["target_tavg"]).reset_index(drop=True)

    for column in list(cleaned.columns):
        if column in SPARSE_OPTIONAL_COLUMNS and column in cleaned.columns:
            missing_share = float(cleaned[column].isna().mean())
            if missing_share >= 0.95:
                cleaned = cleaned.drop(columns=column)
                continue

        if column in cleaned.columns and column not in {"date", "season", "target_tavg", "target_anomaly", "climatic_norm"} and cleaned[column].isna().all():
            cleaned = cleaned.drop(columns=column)

    cleaned = add_calendar_features(cleaned)

    cleaned["climatic_norm"] = np.nan
    cleaned["target_anomaly"] = np.nan

    required_columns = sorted(CORE_REQUIRED_COLUMNS.intersection(cleaned.columns)) + [
        "month",
        "dayofyear",
        "dayofweek",
        "season",
        "season_code",
        "doy_sin",
        "doy_cos",
    ]
    required_columns = [column for column in required_columns if column in cleaned.columns]

    cleaned = cleaned.dropna(subset=required_columns).reset_index(drop=True)

    preferred_order = [
        "date", "target_tavg",
        "tmin", "tmax", "temp_range", "prcp_sum", "snow",
        "pres_mean", "wspd_mean", "wpgt_max", "rhum_mean",
        "dwpt_mean", "tsun_sum", "month", "dayofyear", "dayofweek",
        "season", "season_code", "doy_sin", "doy_cos"
    ]

    ordered_columns = [column for column in preferred_order if column in cleaned.columns]
    return cleaned[ordered_columns]


def build_modeling_dataset(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Полный пайплайн построения датасета для моделирования."""

    selection, candidates, daily, hourly = fetch_meteostat_data(start=start, end=end)
    save_raw_snapshots(hourly=hourly, daily=daily)

    hourly_daily = aggregate_hourly_to_daily(hourly)
    daily_fallback = prepare_daily_fallback(daily)
    combined = combine_hourly_and_daily_features(hourly_daily=hourly_daily, daily_fallback=daily_fallback)
    dataset = apply_missing_value_strategy(combined)

    train, validation, test = split_dataset_by_dates(dataset)

    climatic_norm = compute_train_climatic_norm(train)

    train = apply_climatic_norm(train, climatic_norm)
    validation = apply_climatic_norm(validation, climatic_norm)
    test = apply_climatic_norm(test, climatic_norm)

    dataset_full = pd.concat(
        [train, validation, test],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)

    dataset_full = add_target_history_features(dataset_full)

    return dataset_full, selection, candidates


def get_or_build_modeling_dataset(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Строит датасет из Meteostat, а при неудаче использует локальные raw-снимки."""

    try:
        return build_modeling_dataset(start=start, end=end)
    except Exception as exc:
        daily_frame, hourly_frame = load_raw_snapshots()
        daily_frame, hourly_frame = subset_raw_snapshots(
            daily_frame=daily_frame,
            hourly_frame=hourly_frame,
            start=start,
            end=end,
        )
        hourly_daily = aggregate_hourly_to_daily(hourly_frame)
        daily_fallback = prepare_daily_fallback(daily_frame)
        combined = combine_hourly_and_daily_features(hourly_daily=hourly_daily, daily_fallback=daily_fallback)
        dataset = apply_missing_value_strategy(combined)

        selection = {
            "source_type": "local_raw_csv",
            "station_id": "LOCAL_CACHE",
            "station_name": VOLGOGRAD_NAME,
            "country": "RU",
            "region": "Волгоградская область",
            "distance_km": 0.0,
            "latitude": VOLGOGRAD_LATITUDE,
            "longitude": VOLGOGRAD_LONGITUDE,
            "elevation": VOLGOGRAD_ELEVATION,
            "note": f"Использованы локальные raw-данные из data/raw из-за ошибки сетевой загрузки: {exc}",
        }
        candidates = pd.DataFrame()
        return dataset, selection, candidates


def save_processed_dataset(dataset: pd.DataFrame) -> Path:
    """Сохраняет итоговый датасет в папку data/processed."""

    ensure_project_directories()
    output_path = PROCESSED_DATA_DIR / MODEL_DATASET_FILENAME
    dataset.to_csv(output_path, index=False)
    return output_path


def describe_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу пропусков по столбцам."""

    summary = pd.DataFrame(
        {
            "missing_count": dataset.isna().sum(),
            "missing_share": dataset.isna().mean(),
        }
    )
    summary = summary[summary["missing_count"] > 0].sort_values(["missing_count", "missing_share"], ascending=False)
    return summary


def split_train_validation_test(
    dataset: pd.DataFrame,
    validation_days: int = VALIDATION_DAYS,
    test_days: int = TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит датасет на train, validation и test по времени."""

    if len(dataset) <= validation_days + test_days:
        raise ValueError("Недостаточно наблюдений для выделения validation и test интервалов.")

    train = dataset.iloc[: -(validation_days + test_days)].copy()
    validation = dataset.iloc[-(validation_days + test_days) : -test_days].copy()
    test = dataset.iloc[-test_days:].copy()

    return train, validation, test


def split_dataset_by_dates(
    dataset: pd.DataFrame,
    validation_start: str | pd.Timestamp = VALIDATION_START_DATE,
    validation_end: str | pd.Timestamp = VALIDATION_END_DATE,
    test_start: str | pd.Timestamp = TEST_START_DATE,
    test_end: str | pd.Timestamp = TEST_END_DATE,
    train_start: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит датасет на train, validation и test по фиксированным датам."""

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    validation_start_ts = pd.Timestamp(validation_start).normalize()
    validation_end_ts = pd.Timestamp(validation_end).normalize()
    test_start_ts = pd.Timestamp(test_start).normalize()
    test_end_ts = pd.Timestamp(test_end).normalize()

    if train_start is None:
        train_start_ts = frame["date"].min()
    else:
        train_start_ts = pd.Timestamp(train_start).normalize()

    train_end_ts = validation_start_ts - pd.Timedelta(days=1)

    train = frame.loc[(frame["date"] >= train_start_ts) & (frame["date"] <= train_end_ts)].copy()
    validation = frame.loc[(frame["date"] >= validation_start_ts) & (frame["date"] <= validation_end_ts)].copy()
    test = frame.loc[(frame["date"] >= test_start_ts) & (frame["date"] <= test_end_ts)].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Фиксированные интервалы train / validation / test не удалось сформировать.")

    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def get_interval_summary(frame: pd.DataFrame, label: str) -> dict:
    """Формирует краткую сводку по временному интервалу."""

    return {
        "subset": label,
        "rows": int(len(frame)),
        "start_date": frame["date"].min().date(),
        "end_date": frame["date"].max().date(),
    }


def get_model_feature_columns(dataset: pd.DataFrame, exclude_columns: Iterable[str] | None = None) -> list[str]:
    """Возвращает список числовых признаков для моделей."""

    exclude = {"date", "season", "target_tavg", "target_anomaly"}
    if exclude_columns is not None:
        exclude.update(exclude_columns)

    feature_columns = [
        column
        for column in dataset.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    return sorted(feature_columns)


def get_yearly_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    """Оценивает полноту наблюдений по годам."""

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year

    coverage = frame.groupby("year").agg(rows=("date", "size")).reset_index()
    coverage["days_in_year"] = coverage["year"].apply(
        lambda year: 366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365
    )
    coverage["share_of_full_year"] = coverage["rows"] / coverage["days_in_year"]
    return coverage


def get_max_quality_train_start(
    dataset: pd.DataFrame,
    validation_start: str | pd.Timestamp = VALIDATION_START_DATE,
    min_year_share: float = MAX_QUALITY_MIN_YEAR_SHARE,
) -> pd.Timestamp:
    """Находит самый ранний полный и достаточно качественный год для train."""

    validation_start_ts = pd.Timestamp(validation_start).normalize()
    coverage = get_yearly_coverage(dataset)
    eligible_years = coverage.loc[
        (coverage["year"] < validation_start_ts.year) & (coverage["share_of_full_year"] >= min_year_share),
        "year",
    ]

    if eligible_years.empty:
        return pd.Timestamp(dataset["date"].min()).normalize()

    return pd.Timestamp(f"{int(eligible_years.min())}-01-01")


def build_train_scenarios(
    dataset: pd.DataFrame,
    validation_start: str | pd.Timestamp = VALIDATION_START_DATE,
    candidate_lengths_years: Iterable[int] = TRAIN_LENGTH_YEARS,
) -> list[dict]:
    """Формирует сценарии с разной длиной train при фиксированных validation и test."""

    validation_start_ts = pd.Timestamp(validation_start).normalize()
    dataset_start_ts = pd.Timestamp(dataset["date"].min()).normalize()
    max_quality_start = get_max_quality_train_start(dataset=dataset, validation_start=validation_start_ts)

    scenarios: list[dict] = []
    for years in candidate_lengths_years:
        scenario_start = validation_start_ts - pd.DateOffset(years=years)
        if scenario_start < dataset_start_ts:
            continue
        scenarios.append(
            {
                "scenario": {
                    3: "Короткая история",
                    6: "Средняя история",
                    10: "Длинная история",
                }.get(years, f"История {years} лет"),
                "train_start": scenario_start.normalize(),
                "train_end": validation_start_ts - pd.Timedelta(days=1),
                "description": f"История длиной примерно {years} лет",
            }
        )

    scenarios.append(
        {
            "scenario": "Максимум качественной истории",
            "train_start": max_quality_start.normalize(),
            "train_end": validation_start_ts - pd.Timedelta(days=1),
            "description": "Самая ранняя доступная полная история после проверки годового покрытия",
        }
    )

    deduplicated: list[dict] = []
    seen: set[pd.Timestamp] = set()
    for scenario in sorted(scenarios, key=lambda item: item["train_start"]):
        if scenario["train_start"] in seen:
            continue
        seen.add(scenario["train_start"])
        deduplicated.append(scenario)

    return deduplicated


def build_tabular_modeling_frame(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Готовит табличный массив данных для линейных и нелинейных моделей."""

    frame = dataset.copy().sort_values("date").reset_index(drop=True)

    weather_columns = [
        column
        for column in [
            "tmin",
            "tmax",
            "temp_range",
            "prcp_sum",
            "snow",
            "pres_mean",
            "wspd_mean",
            "wpgt_max",
            "rhum_mean",
            "dwpt_mean",
            "tsun_sum",
        ]
        if column in frame.columns
    ]

    for column in weather_columns:
        frame[f"{column}_lag1"] = frame[column].shift(1)

    calendar_columns = [
        column
        for column in ["month", "dayofyear", "dayofweek", "season_code", "doy_sin", "doy_cos"]
        if column in frame.columns
    ]
    history_columns = [
        column
        for column in [
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_7",
            "lag_14",
            "lag_30",
            "rolling_mean_3",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_std_7",
            "rolling_std_14",
        ]
        if column in frame.columns
    ]
    lagged_weather_columns = [f"{column}_lag1" for column in weather_columns if f"{column}_lag1" in frame.columns]

    feature_columns = calendar_columns + history_columns + lagged_weather_columns
    frame = frame.dropna(subset=feature_columns + ["target_anomaly"]).reset_index(drop=True)
    return frame, feature_columns, weather_columns


def build_sequence_features(dataset: pd.DataFrame) -> list[str]:
    """Возвращает набор признаков для последовательностных нейросетей."""

    return [
        column
        for column in [
            "target_anomaly",
            "tmin",
            "tmax",
            "temp_range",
            "prcp_sum",
            "snow",
            "pres_mean",
            "wspd_mean",
            "rhum_mean",
            "doy_sin",
            "doy_cos",
        ]
        if column in dataset.columns
    ]


def run_adf_test(series: pd.Series, series_name: str) -> pd.DataFrame:
    """Выполняет тест Дики — Фуллера для одного временного ряда."""

    from typing import Any
    from statsmodels.tsa.stattools import adfuller

    clean_series = pd.Series(series, name=series_name).dropna().astype(float)
    if len(clean_series) < 10:
        raise ValueError("Для теста Дики — Фуллера недостаточно наблюдений после удаления пропусков.")

    res_raw: Any = adfuller(clean_series, autolag="AIC")
    
    statistic, p_value, used_lag, observations, critical_values, _ = res_raw
    
    return pd.DataFrame(
        [
            {
                "series": series_name,
                "observations": int(observations),
                "used_lag": int(used_lag),
                "adf_statistic": float(statistic),
                "p_value": float(p_value),
                "critical_value_1pct": float(critical_values["1%"]),
                "critical_value_5pct": float(critical_values["5%"]),
                "critical_value_10pct": float(critical_values["10%"]),
                "stationary_at_5pct": bool(p_value < 0.05),
            }
        ]
    )


def plot_acf_pacf(series: pd.Series, output_path: Path, lags: int = 60) -> Path:
    """Строит графики автокорреляции и частичной автокорреляции."""

    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    clean_series = pd.Series(series).dropna().astype(float)
    if len(clean_series) <= lags + 1:
        lags = max(1, len(clean_series) // 2 - 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(clean_series, lags=lags, ax=axes[0])
    plot_pacf(clean_series, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("Автокорреляционная функция температурных аномалий")
    axes[1].set_title("Частичная автокорреляционная функция температурных аномалий")
    for ax in axes:
        ax.set_xlabel("Лаг, дней")
        ax.set_ylabel("Коэффициент")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def plot_rolling_mean_std(series: pd.Series, output_path: Path, window: int = 30) -> Path:
    """Строит скользящее среднее и скользящее стандартное отклонение."""

    import matplotlib.pyplot as plt

    clean_series = pd.Series(series).dropna().astype(float)
    rolling_mean = clean_series.rolling(window=window).mean()
    rolling_std = clean_series.rolling(window=window).std()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(clean_series.index, clean_series, label="исходный ряд", alpha=0.35)
    axes[0].plot(rolling_mean.index, rolling_mean, label=f"скользящее среднее, {window} дней", linewidth=2)
    axes[0].set_title("Скользящее среднее температурных аномалий")
    axes[0].set_ylabel("Температура, °C")
    axes[0].legend()

    axes[1].plot(rolling_std.index, rolling_std, color="tab:orange", label=f"скользящее стандартное отклонение, {window} дней")
    axes[1].set_title("Скользящее стандартное отклонение температурных аномалий")
    axes[1].set_xlabel("Дата")
    axes[1].set_ylabel("Температура, °C")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def get_ar_lag_specifications() -> list[dict]:
    """Возвращает набор авторегрессионных спецификаций для сравнения."""

    return [
        {"model": "AR(1)", "lags": [1]},
        {"model": "AR(7)", "lags": [7]},
        {"model": "AR(30)", "lags": [30]},
        {"model": "AR с короткими лагами", "lags": [1, 2, 3, 7, 14, 30]},
        {"model": "AR с сезонным лагом", "lags": [1, 2, 3, 7, 14, 30, 365]},
    ]


def fit_autoreg_lags(series: pd.Series | np.ndarray, lags: Iterable[int]):
    """Обучает AutoReg с явно заданными лагами без сезонных фиктивных признаков."""

    from statsmodels.tsa.ar_model import AutoReg

    clean_series = pd.Series(series).dropna().astype(float)
    lag_list = sorted({int(lag) for lag in lags})
    if not lag_list:
        raise ValueError("не задан ни один лаг")

    max_lag = max(lag_list)
    if len(clean_series) <= max_lag + 1:
        raise ValueError(f"недостаточно наблюдений для лага {max_lag}")

    return AutoReg(clean_series.to_numpy(), lags=lag_list, old_names=False, seasonal=False).fit()


def forecast_autoreg_lags(
    history_series: pd.Series | np.ndarray,
    horizon: int,
    lags: Iterable[int],
) -> np.ndarray:
    """Строит многошаговый прогноз AutoReg на весь горизонт.

    Эта функция оставлена для технических проверок и не используется
    в основном сравнении моделей.
    """

    clean_series = pd.Series(history_series).dropna().astype(float)
    model = fit_autoreg_lags(history_series, lags)
    forecast = model.predict(start=len(clean_series), end=len(clean_series) + horizon - 1, dynamic=False)
    return np.asarray(forecast, dtype=float)


def forecast_autoreg_walk_forward(
    history_series: pd.Series | np.ndarray,
    target_series: pd.Series | np.ndarray,
    lags: Iterable[int],
) -> np.ndarray:
    """Строит последовательный однодневный прогноз AutoReg."""

    history = list(pd.Series(history_series).dropna().astype(float))
    targets = pd.Series(target_series).astype(float)
    lag_list = sorted({int(lag) for lag in lags})
    predictions: list[float] = []

    if not lag_list:
        return np.full(len(targets), np.nan, dtype=float)

    max_lag = max(lag_list)
    for actual_value in targets:
        if len(history) <= max_lag + 1:
            predictions.append(np.nan)
        else:
            try:
                model = fit_autoreg_lags(pd.Series(history), lag_list)
                forecast = model.predict(start=len(history), end=len(history), dynamic=False)
                predictions.append(float(np.asarray(forecast)[0]))
            except Exception:
                predictions.append(np.nan)

        if pd.notna(actual_value):
            history.append(float(actual_value))

    return np.asarray(predictions, dtype=float)


def select_autoreg_lag_models(
    series: pd.Series | np.ndarray,
    lag_specs: list[dict] | None = None,
) -> pd.DataFrame:
    """Сравнивает авторегрессионные спецификации по AIC и BIC."""

    specs = get_ar_lag_specifications() if lag_specs is None else lag_specs
    rows = []

    for spec in specs:
        model_name = spec["model"]
        lag_list = [int(lag) for lag in spec["lags"]]
        try:
            result = fit_autoreg_lags(series, lag_list)
            rows.append(
                {
                    "model": model_name,
                    "lags": ", ".join(str(lag) for lag in lag_list),
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "nobs": int(result.nobs),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "lags": ", ".join(str(lag) for lag in lag_list),
                    "aic": np.nan,
                    "bic": np.nan,
                    "nobs": int(pd.Series(series).dropna().shape[0]),
                    "status": str(exc),
                }
            )

    return pd.DataFrame(rows, columns=["model", "lags", "aic", "bic", "nobs", "status"])


def create_sequence_windows(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = SEQUENCE_WINDOW_DAYS,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Формирует окна временных последовательностей для GRU-модели."""
    frame = dataset.copy().sort_values("date").reset_index(drop=True)
    values = frame[feature_columns].to_numpy(dtype=float)
    
    targets = frame["target_anomaly"].to_numpy(dtype=float) 
    dates = pd.to_datetime(frame["date"])

    windows: list[np.ndarray] = []
    window_targets: list[float] = []
    window_dates: list[pd.Timestamp] = []

    for idx in range(window_size, len(frame)):
        windows.append(values[idx - window_size : idx])
        window_targets.append(targets[idx])
        window_dates.append(dates.iloc[idx])

    return np.asarray(windows, dtype=float), np.asarray(window_targets, dtype=float), pd.Series(window_dates, name="date")


def split_sequence_windows_by_dates(
    windows: np.ndarray,
    targets: np.ndarray,
    target_dates: pd.Series,
    train_start: str | pd.Timestamp,
    validation_start: str | pd.Timestamp = VALIDATION_START_DATE,
    validation_end: str | pd.Timestamp = VALIDATION_END_DATE,
    test_start: str | pd.Timestamp = TEST_START_DATE,
    test_end: str | pd.Timestamp = TEST_END_DATE,
) -> dict[str, np.ndarray | pd.Series]:
    """Делит последовательностные окна по тем же фиксированным датам."""

    date_index = pd.to_datetime(target_dates)
    train_start_ts = pd.Timestamp(train_start).normalize()
    validation_start_ts = pd.Timestamp(validation_start).normalize()
    validation_end_ts = pd.Timestamp(validation_end).normalize()
    test_start_ts = pd.Timestamp(test_start).normalize()
    test_end_ts = pd.Timestamp(test_end).normalize()

    train_mask = (date_index >= train_start_ts) & (date_index < validation_start_ts)
    validation_mask = (date_index >= validation_start_ts) & (date_index <= validation_end_ts)
    test_mask = (date_index >= test_start_ts) & (date_index <= test_end_ts)

    return {
        "X_train": windows[train_mask],
        "y_train": targets[train_mask],
        "dates_train": date_index[train_mask].reset_index(drop=True),
        "X_validation": windows[validation_mask],
        "y_validation": targets[validation_mask],
        "dates_validation": date_index[validation_mask].reset_index(drop=True),
        "X_test": windows[test_mask],
        "y_test": targets[test_mask],
        "dates_test": date_index[test_mask].reset_index(drop=True),
    }


def calculate_mase(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    insample: pd.Series | np.ndarray,
    seasonality: int = SEASONAL_PERIOD_DAYS,
) -> float:
    """Вычисляет MASE относительно сезонного наивного масштаба."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    insample_array = np.asarray(insample, dtype=float)

    valid_mask = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    y_true_array = y_true_array[valid_mask]
    y_pred_array = y_pred_array[valid_mask]
    insample_array = insample_array[np.isfinite(insample_array)]

    effective_seasonality = seasonality if len(insample_array) > seasonality else 1
    naive_errors = np.abs(insample_array[effective_seasonality:] - insample_array[:-effective_seasonality])

    scale = float(np.mean(naive_errors)) if len(naive_errors) > 0 else np.nan
    if np.isnan(scale) or scale == 0:
        return float("nan")

    return float(np.mean(np.abs(y_true_array - y_pred_array)) / scale)


def seasonal_naive_forecast(
    target_dates: pd.Series | pd.DatetimeIndex,
    full_series: pd.Series,
) -> pd.Series:
    """Строит seasonal naive прогноз по значению примерно того же дня прошлого года."""

    history = full_series.copy()
    history.index = pd.to_datetime(history.index)
    target_index = pd.to_datetime(target_dates)

    predictions: list[float] = []
    for current_date in target_index:
        candidate_dates = [current_date - pd.DateOffset(years=1)]

        if current_date.month == 2 and current_date.day == 29:
            candidate_dates.append(pd.Timestamp(year=current_date.year - 1, month=2, day=28))

        candidate_dates.extend(
            [
                current_date - pd.DateOffset(years=1) - pd.Timedelta(days=1),
                current_date - pd.DateOffset(years=1) + pd.Timedelta(days=1),
            ]
        )

        prediction_value = np.nan
        for candidate_date in candidate_dates:
            normalized_candidate = pd.Timestamp(candidate_date).normalize()
            if normalized_candidate in history.index:
                prediction_value = float(history.loc[normalized_candidate])
                break

        predictions.append(prediction_value)

    return pd.Series(predictions, index=target_index, name="seasonal_naive")


def evaluate_forecast(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    insample: pd.Series | np.ndarray,
    seasonality: int = SEASONAL_PERIOD_DAYS,
) -> dict:
    """Считает набор метрик прогноза, включая MASE."""

    base_metrics = evaluate_regression(y_true=y_true, y_pred=y_pred, model_name=model_name)
    base_metrics["mase"] = calculate_mase(
        y_true=y_true,
        y_pred=y_pred,
        insample=insample,
        seasonality=seasonality,
    )
    return base_metrics


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Фиксирует основные генераторы случайных чисел."""

    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def get_default_runtime_config() -> dict:
    """Возвращает базовую конфигурацию экспериментов."""

    return {
        "data_start_date": pd.Timestamp(DEFAULT_START_DATE),
        "data_end_date": pd.Timestamp(FIXED_DATA_END_DATE),
        "validation_start_date": pd.Timestamp(VALIDATION_START_DATE),
        "validation_end_date": pd.Timestamp(VALIDATION_END_DATE),
        "test_start_date": pd.Timestamp(TEST_START_DATE),
        "test_end_date": pd.Timestamp(TEST_END_DATE),
        "seasonal_period_days": SEASONAL_PERIOD_DAYS,
        "sequence_window_days": SEQUENCE_WINDOW_DAYS,
        "neural_epochs": NEURAL_EPOCHS,
        "neural_patience": NEURAL_PATIENCE,
        "neural_batch_size": NEURAL_BATCH_SIZE,
        "random_state": RANDOM_STATE,
    }


def evaluate_regression(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, model_name: str) -> dict:
    """Вычисляет основные метрики регрессии."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    valid_mask = np.isfinite(y_true_array) & np.isfinite(y_pred_array)

    if not np.any(valid_mask):
        return {
            "model": model_name,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
        }

    y_true_array = y_true_array[valid_mask]
    y_pred_array = y_pred_array[valid_mask]

    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "bias": float(np.mean(y_pred_array - y_true_array)),
    }

def compute_train_climatic_norm(
    train_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Вычисляет климатическую норму только по train-выборке."""

    if "dayofyear" not in train_frame.columns:
        raise ValueError("В train_frame отсутствует столбец dayofyear.")

    climatic_norm = (
        train_frame.groupby("dayofyear", as_index=False)["target_tavg"]
        .mean()
        .rename(columns={"target_tavg": "climatic_norm"})
    )

    return climatic_norm


def apply_climatic_norm(
    frame: pd.DataFrame,
    climatic_norm: pd.DataFrame,
) -> pd.DataFrame:
    """Добавляет климатическую норму и температурные аномалии."""

    enriched = frame.copy()

    enriched = enriched.merge(
        climatic_norm,
        on="dayofyear",
        how="left",
    )

    enriched["target_anomaly"] = (
        enriched["target_tavg"] - enriched["climatic_norm"]
    )

    return enriched