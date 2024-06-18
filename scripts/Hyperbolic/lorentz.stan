functions {
    real hyp(real t1, real t2, vector E1, vector E2){
        return acosh(t1*t2 - dot_product(E1, E2));
    }
}
data {
    int<lower=0> N;        // number of points
    int<lower=0> D;        // Dimension of space
    matrix[N, N] deltaij;  // matrix of data distances, only upper triangle will be operated on
}
transformed data {
    real Nterms = 0.5*N*(N-1);
}
parameters {
    vector[D] euc[N];                // directions
    vector<lower=0.0>[N] sig;
    real<lower=0.001> lambda;
}
transformed parameters {
    vector[N] time;
    
    for (i in 1:N)
        time[i] = sqrt(1.0 + dot_self(euc[i]));
}
model {
    real dist;
    real seff;
    
    target += -Nterms*0.5*square(lambda)/square(10.0);
      
    for (i in 1:N)
        sig[i] ~ inv_gamma(2.0, 0.5);
    
    for(i in 1:N){
        for (j in i+1:N){
            if (deltaij[i,j] > 0.0){
                dist = hyp(time[i], time[j], euc[i], euc[j]);
                seff = sqrt(square(sig[i]) + square(sig[j]));

                deltaij[i,j] ~ normal(dist/lambda, seff);
            }
        }
    }
}