import os
import sys
import glob
from mcp.server.fastmcp import FastMCP
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# --- WINDOWS CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HADOOP_BIN_DIR = os.path.join(BASE_DIR, 'hadoop', 'bin')
os.environ['HADOOP_HOME'] = os.path.join(BASE_DIR, 'hadoop')

# Add hadoop/bin to PATH for hadoop.dll
if HADOOP_BIN_DIR not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + HADOOP_BIN_DIR

# Initialize Server
mcp = FastMCP("RedditEchoChamberAnalytics")

# --- GLOBAL STATE ---
spark = None
title_model = None
comment_model = None
posts_df = None
gdelt_df = None

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data_cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GDELT_DIR = os.path.join(BASE_DIR, "gdelt_data")

def log(msg):
    """Helper to print to stderr so we don't break MCP protocol"""
    print(msg, file=sys.stderr)

def initialize_system():
    """
    EAGER LOADING: Starts Spark and loads ALL data/models into RAM.
    """
    global spark, title_model, comment_model, posts_df, gdelt_df
    
    log("[START] SYSTEM INITIALIZATION STARTED...")
    
    # 1. Start Spark
    log("[INFO] [1/5] Starting Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("RedditMCPServer") \
            .master("local[4]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.ui.showConsoleProgress", "false") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    except Exception as e:
        log(f"[ERROR] CRITICAL SPARK ERROR: {e}")
        return

    # 2. Load Models
    log("[INFO] [2/5] Loading ML Models...")
    try:
        # Check if model directories exist first
        lr_path = os.path.join(MODELS_DIR, "spark_lr_model")
        comments_path = os.path.join(MODELS_DIR, "spark_lr_comments_model")
        
        if os.path.exists(lr_path):
            title_model = PipelineModel.load(lr_path)
        else:
            log(f"[WARN] Title model not found at {lr_path}")

        if os.path.exists(comments_path):
            comment_model = PipelineModel.load(comments_path)
        else:
            log(f"[WARN] Comment model not found at {comments_path}")
            
    except Exception as e:
        log(f"[WARN] Warning: Could not load models. Error: {e}")

    # 3. Load & Cache Reddit Data
    log("[INFO] [3/5] Caching Reddit Data...")
    try:
        # Use Python glob to find files, avoiding Spark wildcard issues on Windows
        posts_pattern = os.path.join(DATA_DIR, "*_posts.csv")
        posts_files = glob.glob(posts_pattern)
        
        if posts_files:
            log(f"   Found {len(posts_files)} post files.")
            posts_df = spark.read \
                .option("header", True) \
                .option("inferSchema", False) \
                .option("encoding", "UTF-8") \
                .csv(posts_files) # Pass the list of specific files
            
            # Safe casting
            posts_df = posts_df.withColumn("score", F.expr("try_cast(score as int)"))
            posts_df.cache()
            posts_df.count() # Force materialization
        else:
            log(f"[ERROR] No post files found matching: {posts_pattern}")
            
    except Exception as e:
        log(f"[ERROR] Error loading Reddit data: {e}")

    # 4. Load & Cache GDELT Data
    log("[INFO] [4/5] Caching GDELT Data...")
    try:
        gdelt_pattern = os.path.join(GDELT_DIR, "*.CSV")
        gdelt_files = glob.glob(gdelt_pattern)
        
        if gdelt_files:
            log(f"   Found {len(gdelt_files)} GDELT files.")
            # Read without schema (robust method)
            raw_gdelt = spark.read.csv(gdelt_files, sep='\t', inferSchema=False)
            
            # Select and rename columns by index (SQLDATE=1, AvgTone=34)
            gdelt_df = raw_gdelt.select(
                F.col("_c1").alias("SQLDATE"),
                F.col("_c34").cast(DoubleType()).alias("AvgTone")
            ).withColumn("date", F.to_date(F.col("SQLDATE"), "yyyyMMdd"))
            
            gdelt_df.cache()
            gdelt_df.count() # Force materialization
        else:
            log(f"[WARN] No GDELT data found in {GDELT_DIR}")
    except Exception as e:
        log(f"[ERROR] Error loading GDELT data: {e}")

    log("[SUCCESS] [5/5] SYSTEM READY! Server is listening.")

# --- TOOLS ---

@mcp.tool()
def predict_subreddit_from_title(title: str) -> str:
    """Predicts subreddit from a post title."""
    if not title_model: return "Model not loaded."
    try:
        data = [(title, "dummy")]
        df = spark.createDataFrame(data, ["title", "subreddit"])
        result = title_model.transform(df).select("predicted_subreddit").first()[0]
        return f"Predicted Subreddit: r/{result}"
    except Exception as e:
        return f"Error predicting: {e}"

@mcp.tool()
def predict_subreddit_from_comment(comment: str) -> str:
    """Predicts subreddit from a comment text."""
    if not comment_model: return "Model not loaded."
    try:
        data = [(comment, "dummy")]
        df = spark.createDataFrame(data, ["text", "subreddit"])
        result = comment_model.transform(df).select("predicted_subreddit").first()[0]
        return f"Predicted Subreddit: r/{result}"
    except Exception as e:
        return f"Error predicting: {e}"

@mcp.tool()
def get_subreddit_stats(subreddit: str) -> str:
    """Get post count and avg score for a subreddit."""
    if posts_df is None: return "Data not loaded."
    try:
        stats = posts_df.filter(F.col("subreddit") == subreddit).agg(
            F.count("*").alias("count"),
            F.avg("score").alias("avg_score")
        ).first()
        return f"r/{subreddit}: {stats['count']:,} posts, Avg Score: {stats['avg_score']:.2f}"
    except Exception as e:
        return f"Error getting stats: {e}"

@mcp.tool()
def compare_keyword_frequency(keyword: str) -> str:
    """Compare keyword frequency in politics vs Conservative."""
    if posts_df is None: return "Data not loaded."
    try:
        mentions = posts_df.filter(F.col("subreddit").isin(["politics", "Conservative"])) \
            .groupBy("subreddit") \
            .agg(F.sum(F.when(F.contains(F.lower(F.col("title")), keyword.lower()), 1).otherwise(0)).alias("mentions")) \
            .collect()
        results = {row['subreddit']: row['mentions'] for row in mentions}
        return f"'{keyword}': politics={results.get('politics',0)}, Conservative={results.get('Conservative',0)}"
    except Exception as e:
        return f"Error comparing keywords: {e}"

@mcp.tool()
def get_global_news_sentiment(date_str: str) -> str:
    """Gets average global news sentiment (AvgTone) for a date (YYYY-MM-DD)."""
    if gdelt_df is None: return "GDELT data not available."
    try:
        day_stats = gdelt_df.filter(F.col("date") == date_str).agg(
            F.avg("AvgTone").alias("avg_tone"),
            F.count("*").alias("count")
        ).first()
        
        if not day_stats or day_stats['count'] == 0:
            return f"No news data found for {date_str}."
            
        tone = day_stats['avg_tone']
        sentiment = "Positive" if tone > 0 else "Negative"
        return f"Global News ({date_str}): Avg Sentiment {tone:.2f} ({sentiment}), Events: {day_stats['count']:,}"
    except Exception as e:
        return f"Error getting GDELT data: {e}"

if __name__ == "__main__":
    # Initialize logic
    initialize_system()
    # Run MCP
    mcp.run()