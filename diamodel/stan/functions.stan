functions {
  // Response curve as Generalized gamma distribution with scale=1
  // https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gengamma.html
  vector curve(vector x, real peak, real tail) {
    real a = tail / exp(lgamma(peak));
    vector[size(x)] y = a * (x .^ (peak * tail - 1)) .* exp(-(x .^ tail));
    return y;
  }
  
  // Combined curve for multiple overlapping curves with different params.
  // xg: (ncurves, nticks) - x-grid values for each curve
  // amounts: (ncurves,) - amount for each curve
  // peaks: (ncurves,) - peak value for each curve
  // Returns: (nticks,) - combined curve values
  vector combined_curve(matrix xg, vector amounts, vector peaks) {
    int ncurves = size(amounts);
    int nticks = cols(xg);
    
    vector[nticks] resp = rep_vector(0.0, nticks);
    
    for (ic in 1 : ncurves) {
      resp += amounts[ic] * curve(to_vector(xg[ic]), peaks[ic], 1.0);
    }
    
    return resp;
  }
}
