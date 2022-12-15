from pyspark.sql.functions import length, col
import logging


def remove_outliers(
    spark,
    database_name="insuranceqa",
    split: str = "train",
    suffix: str = "silver",
    mode: str = "overwrite",
    filter: str = "length < 50",
):
    """Remove utterances that are longer than the 75th quantile."""

    query = f"SELECT * FROM {database_name}.{split}"
    logging.info(f"Select query for DF: {query}")
    df = spark.sql(query)
    logging.info(f"Dataframe contains {df.count()} before cleaning")
    df = (
        df.withColumn("length", length(col("question_en")))
        .filter(filter)  # Limit size to approx. 75th quantile
        .drop("length")
    )
    target_table = f"{database_name}.{split}_{suffix}"
    logging.info(f"Writing clean dataframe into")
    logging.info(f"{df.count()} after cleaning")
    df.write.saveAsTable(target_table, mode=mode)
    logging.info(f"Success")
