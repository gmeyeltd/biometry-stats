# Step 3a: runtime of the six WHWK functions (unmodified Rallfun v45, default nboot), datasets run one after another.
# Writes outputs/benchmark_R.csv (elapsed seconds per function and dataset).
get_script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a))) else getwd()
}
BASE    <- get_script_dir()
RALLFUN <- normalizePath(file.path(BASE, "..", "Rallfun-v45.txt"))
OUTDIR  <- file.path(BASE, "outputs"); dir.create(OUTDIR, showWarnings = FALSE)
ds      <- read.csv(file.path(BASE, "prepared", "datasets.csv"), stringsAsFactors = FALSE)

bench_one <- function(did) {
  quiet <- function(expr) { val <- NULL; utils::capture.output(val <- expr); val }
  quiet(source(RALLFUN))
  x <- as.matrix(read.csv(file.path(BASE, "prepared", paste0(did, ".csv")), check.names = FALSE))
  tm <- function(expr) system.time(quiet(expr))[["elapsed"]]
  r <- data.frame(dataset = did, n = nrow(x),
    MAE     = tm(oph.dep.comMeanAE(x, invalid = 6)),
    RMSE    = tm(oph.dep.comRMSAE(x, invalid = 6)),
    MPE     = tm(oph.dep.commean(x, invalid = 6)),
    MedAE   = tm(oph.dep.comMedAE(x, invalid = 6)),
    SD      = tm(oph.dep.comvar(x, invalid = 6)),
    McNemar = tm(oph.mcnemar(as.data.frame(x), invalid = 6)))
  r$total <- rowSums(r[, c("MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar")])
  r
}

# one dataset at a time, so each R function runs alone on an otherwise idle machine (R is single-threaded)
res <- NULL
for (d in ds$id) { r <- bench_one(d); print(r); res <- rbind(res, r) }
res$R_version <- R.version.string
res$cpu <- Sys.getenv("PROCESSOR_IDENTIFIER")
write.csv(res, file.path(OUTDIR, "benchmark_R.csv"), row.names = FALSE)
print(res)
