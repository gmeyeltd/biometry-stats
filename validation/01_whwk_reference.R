# Step 1: WHWK reference run with UNMODIFIED Rallfun v45, plus export of the bootstrap indices R used.
#
# Every WHWK bootstrap routine (ydbt for MAE/RMSE/MPE; dmedpb for MedAE) calls set.seed(2) and then
#   data <- matrix(sample(n, size = n*nboot, replace = TRUE), nrow = nboot)
# so the resampling matrix is deterministic. We regenerate it here with the same call and save it,
# and check (first pair of each dataset) that it reproduces R's own p-value.
#
# Reads  prepared/*.csv, ../Rallfun-v45.txt
# Writes outputs/R_results.csv, outputs/R_teststats.csv, outputs/R_index_check.csv,
#        outputs/indices/<id>_B1999.csv.gz, outputs/indices/<id>_B599.csv.gz, outputs/R_session.txt

get_script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a))) else getwd()
}
BASE    <- get_script_dir()
RALLFUN <- normalizePath(file.path(BASE, "..", "Rallfun-v45.txt"))
OUTDIR  <- file.path(BASE, "outputs"); dir.create(file.path(OUTDIR, "indices"), recursive = TRUE, showWarnings = FALSE)
ds      <- read.csv(file.path(BASE, "prepared", "datasets.csv"), stringsAsFactors = FALSE)
FORM    <- c("HQ", "H1", "Haigis", "SRK/T", "BU2", "K6")
INVALID <- 6          # as in the original WHWK runs; no |PE| > 4 D exists in these data

run_one <- function(did) {
  quiet <- function(expr) { val <- NULL; utils::capture.output(val <- expr); val }
  quiet(source(RALLFUN))
  x <- as.matrix(read.csv(file.path(BASE, "prepared", paste0(did, ".csv")), check.names = FALSE))
  colnames(x) <- FORM
  n <- nrow(x); J <- ncol(x)
  t0 <- Sys.time()

  # --- the six WHWK functions, default arguments -------------------------------------------
  mae   <- quiet(oph.dep.comMeanAE(x, invalid = INVALID))   # ydbt, nboot 1999
  rmse  <- quiet(oph.dep.comRMSAE(x,  invalid = INVALID))   # ydbt on squares, nboot 1999
  mpe   <- quiet(oph.dep.commean(x,   invalid = INVALID))   # ydbt, default nboot 599
  medae <- quiet(oph.dep.comMedAE(x,  invalid = INVALID))   # dmedpb percentile, nboot 1999
  sdv   <- quiet(oph.dep.comvar(x,    invalid = INVALID))   # Morgan-Pitman / HC4, no bootstrap
  mcn   <- quiet(oph.mcnemar(as.data.frame(x), invalid = INVALID))  # 8 thresholds, Holm

  long <- function(m, metric, thr = NA) data.frame(
    dataset = did, metric = metric, threshold = thr,
    f1 = FORM[m[, 1]], f2 = FORM[m[, 2]], est1 = m[, 3], est2 = m[, 4],
    p = m[, 6], p_adj = m[, 7], stringsAsFactors = FALSE)
  res <- rbind(long(mae, "MAE"), long(rmse, "RMSE"), long(mpe, "MPE"), long(medae, "MedAE"), long(sdv, "SD"))
  for (L in seq_along(mcn)) {
    m <- mcn[[L]]
    res <- rbind(res, data.frame(dataset = did, metric = "McNemar", threshold = m[, 1],
      f1 = FORM[m[, 2]], f2 = FORM[m[, 5]], est1 = m[, 4], est2 = m[, 7],
      p = m[, 8], p_adj = m[, 9], stringsAsFactors = FALSE))
  }

  # --- observed test statistics (before the p-value transform) -----------------------------
  ts <- NULL
  for (j in 1:(J - 1)) for (k in (j + 1):J) {
    ts <- rbind(ts, data.frame(dataset = did, f1 = FORM[j], f2 = FORM[k],
      t_MAE  = yuend(abs(x[, j]), abs(x[, k]), tr = 0)$teststat,
      t_RMSE = yuend(x[, j]^2, x[, k]^2, tr = 0)$teststat,
      t_MPE  = yuend(x[, j], x[, k], tr = 0)$teststat,
      t_SD   = quiet(comdvar(x[, j], x[, k]))$test.stat,
      stringsAsFactors = FALSE))
  }

  # --- regenerate and save the bootstrap index matrices --------------------------------------
  set.seed(2); idx1999 <- matrix(sample(n, size = n * 1999, replace = TRUE), nrow = 1999)
  set.seed(2); idx599  <- matrix(sample(n, size = n * 599,  replace = TRUE), nrow = 599)
  write.table(idx1999, gzfile(file.path(OUTDIR, "indices", paste0(did, "_B1999.csv.gz"))), sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(idx599,  gzfile(file.path(OUTDIR, "indices", paste0(did, "_B599.csv.gz"))),  sep = ",", row.names = FALSE, col.names = FALSE)

  # --- check: saved indices reproduce R's own p-values (first pair, MAE and MedAE) ---------
  a <- abs(x[, 1]); b <- abs(x[, 2])
  bvec <- apply(idx1999, 1, tsub, a - mean(a), b - mean(b), 0)
  p_mae_manual <- sum(abs(yuend(a, b, tr = 0)$teststat) <= abs(bvec)) / 1999
  psi <- apply(idx1999, 1, function(i) median(a[i]) - median(b[i]))
  tt <- mean(psi > 0) + 0.5 * mean(psi == 0)
  p_medae_manual <- 2 * min(tt, 1 - tt)
  chk <- data.frame(dataset = did, n = n,
    p_MAE_function = mae[1, 6], p_MAE_from_saved_indices = p_mae_manual,
    p_MedAE_function = medae[1, 6], p_MedAE_from_saved_indices = p_medae_manual,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  list(res = res, ts = ts, chk = chk)
}

library(parallel)
ncore <- min(nrow(ds), max(1, detectCores() - 2))
cat(sprintf("Running %d datasets on %d cores...\n", nrow(ds), ncore))
cl <- makeCluster(ncore)
clusterExport(cl, c("BASE", "RALLFUN", "OUTDIR", "FORM", "INVALID"))
out <- parLapply(cl, ds$id, run_one)
stopCluster(cl)

write.csv(do.call(rbind, lapply(out, `[[`, "res")), file.path(OUTDIR, "R_results.csv"), row.names = FALSE)
write.csv(do.call(rbind, lapply(out, `[[`, "ts")),  file.path(OUTDIR, "R_teststats.csv"), row.names = FALSE)
chk <- do.call(rbind, lapply(out, `[[`, "chk"))
write.csv(chk, file.path(OUTDIR, "R_index_check.csv"), row.names = FALSE)
writeLines(c(capture.output(sessionInfo()), "", paste("RNGkind:", paste(RNGkind(), collapse = ", ")),
             paste("Rallfun file:", RALLFUN)), file.path(OUTDIR, "R_session.txt"))
print(chk)
if (any(chk$p_MAE_function != chk$p_MAE_from_saved_indices | chk$p_MedAE_function != chk$p_MedAE_from_saved_indices))
  stop("Saved indices do NOT reproduce R's p-values - stop and report.")
cat("R reference run complete; saved indices reproduce R's own p-values.\n")
