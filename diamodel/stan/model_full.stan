#include functions.stan
data {
  int maxpred; // max prediction horizon, ticks
  int maxc; // max number of overlapping curves
  real gscale; // scale to convert from ticks to gamma distribution domain
  
  int na; // number of anchors, prediction chunks
  array[na] int chunk; // size of each prediction chunk
  
  // stacked insulin curves within each chunk
  int ni; // max number of overlapping curves
  vector[ni] insulin; // insulin amounts
  array[na, 2] int islice; // insulin curves slices (start, ni) for each chunk
  array[na] matrix[maxc, maxpred] ixg; // insulin curves gamma domain
  
  // stacked carbs curves within each chunk
  int nc; // number of carb curves
  vector[nc] carbs; // carbs amounts
  array[na, 2] int cslice; // carbs curves slices (start, nc) for each chunk
  array[na] matrix[maxc, maxpred] cxg; // carb curves gamma domain
  
  // cgm measurements in each chunk
  array[na] vector[maxpred + 1] cgm;
  
  // prior on global unconstrained params
  int<lower=1> nz; // number of unconstrained params
  vector[nz] mu_z; // `z` mean
  matrix[nz, nz] L_z; // `z` covariance as Cholesky factor
}
transformed data {
  vector[na] cgm0;
  for (ia in 1 : na) {
    cgm0[ia] = cgm[ia][1];
  }
}
parameters {
  // global unconstrained params
  vector[nz] z;
  // local (carbs) params
  vector[nc] cpeakzi; // i-th carbs curve peak, non-centered
  vector[nc] ccorri; // i-th carbs curve amount correction
  vector<lower=2>[na] bg0i; // 'true' BG value at the start of each chunk
}
transformed parameters {
  vector[nz] x = exp(z); // unconstrained z to positively constrained x
  
  vector[nc] cpeaki = x[6] + 1.0 * cpeakzi; // centered cpeaki with fixed std
  // clip cpeaki so gamma curve has a reasonable shape and fits maxact interval
  cpeaki = fmax(cpeaki, 1.6);
  cpeaki = fmin(cpeaki, 7.0);
}
model {
  // unpack global params
  real isens = x[1]; // insulin sensitivity
  real ipeak = x[2]; // insulin curve peak
  real sigma = x[3]; // sigma, cgm observation sigma
  real rho = x[4]; // cgm noise correlation coefficient
  real csens = x[5]; // carbs sensitivity
  real cpeak = x[6]; // group mean of carbs curve peaks
  
  // priors
  z ~ multi_normal_cholesky(mu_z, L_z); // joint Gaussian for unconstrained params 
  cpeakzi ~ std_normal();
  ccorri ~ normal(0, 3);
  bg0i ~ normal(cgm0, sigma);
  
  // likelihood by looping through all chunks and rolling out the model predictions
  for (ia in 1 : na) {
    int dpred = chunk[ia]; // prediction duration
    
    // insulin absorption rate (iar)
    int ss = islice[ia][1]; // segment start, first curve id
    int sn = islice[ia][2]; // segment size, num curves
    vector[sn] insulin_ = segment(insulin, ss, sn);
    vector[sn] ipeaks = rep_vector(ipeak, sn);
    vector[dpred] iar = combined_curve(ixg[ia][ : sn], insulin_, ipeaks)[ : dpred];
    iar *= isens * gscale; // iar in bg unit
    
    // carbs absorption rate (car)
    ss = cslice[ia][1]; // segment start, first curve id
    sn = cslice[ia][2]; // segment size, num curves
    vector[sn] carbs_ = segment(carbs, ss, sn) + segment(ccorri, ss, sn);
    carbs_ = fmax(carbs_, 0.0); // carbs can't be negative
    vector[sn] cpeaks = segment(cpeaki, ss, sn);
    vector[dpred] car = combined_curve(cxg[ia][ : sn], carbs_, cpeaks)[ : dpred];
    car *= csens * gscale; // car in bg unit
    
    // // DEBUG
    // // Check for infinite values in car or iar
    // for (i in 1 : dpred) {
    //   if (is_inf(car[i])) {
    //     reject("car[i]=inf i=", i, " anchor=", ia);
    //   }
    //   if (is_inf(iar[i])) {
    //     reject("iar[i]=inf i=", i, " anchor=", ia);
    //   }
    // }
    
    vector[dpred] current_cgm = segment(cgm[ia], 1, dpred);
    vector[dpred] next_cgm = segment(cgm[ia], 2, dpred);
    
    // // Option 1. Predict next BG value    
    // vector[dpred] diff = next_cgm - current_cgm;
    // diff ~ normal(car - iar, sigma);
    
    // // Option 2. Predict BG rollout curve with Gaussian noise
    // vector[dpred] ibg = cumulative_sum(iar); // cumulative insulin in bg
    // vector[dpred] cbg = cumulative_sum(car); // cumulative carbs in bg
    
    // vector[dpred] pred_bg = bg0i[ia] + cbg - ibg;
    // next_cgm ~ normal(pred_bg, sigma);
    
    // Option 3. Predict BG rollout curve with autocorrelated AR(1) noise
    vector[dpred] ibg = cumulative_sum(iar); // cumulative insulin in bg
    vector[dpred] cbg = cumulative_sum(car); // cumulative carbs in bg
    vector[dpred] pred_bg = bg0i[ia] + cbg - ibg;
    
    real res0 = cgm0[ia] - bg0i[ia];
    vector[dpred + 1] res = append_row(res0, next_cgm - pred_bg); // residuals
    
    real res_noise = sigma * sqrt(1 - rho ^ 2);
    res[2 : ] ~ student_t(4, rho * res[1 : dpred], res_noise);
  }
}
