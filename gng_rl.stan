// Go No Go RL full model
// Utility function: RW + Constant Pav + go bias
// Decision model: Softmax + e-greedy
// adopted from HbayesDM

data {
  int<lower=1> ns; // subject number
  int<lower=1> nt; // trial number
  int<lower=1, upper=4> cue[ns, nt]; // cue type
  int<lower=0, upper=3> choice[ns, nt]; // choice data  1 for no go response, 2 for go response
  int<lower=-1, upper=1> val[ns,nt]; // valence of the stimulus, 1 for reward, -1 for punishment
  real outcome[ns, nt]; // outcome for each action
}

transformed data {
  vector[4] initV;
  initV = rep_vector(0.0, 4);
}

parameters {
  vector[5] mu_pr; // group-level mean hyper parameter
  vector<lower=0>[5] sigma; // group-level standard deviation 
  vector[ns] alpha_raw;   // learning rate
  vector[ns] tau_raw; // inverse temperature in softmax
  vector[ns] bias_raw;    // go bias 
  vector[ns] pi_raw;      // pavolovian bias
  vector[ns] ep_raw;       // e-greedy(irreducible noise)
}

transformed parameters {
  vector<lower=0, upper=1>[ns] alpha; // learning rate
  vector<lower=0, upper=1>[ns] ep;    // e-greedy
  vector[ns] bias;                    // go bias
  vector[ns] pi_;                      // pavolovian bias
  vector<lower=-mu_pr[5]/sigma[5]> [ns] tau;           // strictly positive inverse temperature
  matrix[ns,nt] utility;              // matrix for storing softmax output
  
  
  // matt-trick
  alpha = inv_logit(mu_pr[1] + sigma[1] * alpha_raw);
  ep = inv_logit(mu_pr[2] + sigma[2] * ep_raw);
  bias = mu_pr[3] + sigma[3] * bias_raw;
  pi_  = mu_pr[4] + sigma[4] * pi_raw;
  tau = mu_pr[5] + sigma[5] * tau_raw;
  
   // subject loop
  for (i in 1:ns) {
    vector[4] wv_g;  // action weight for go
    vector[4] wv_ng; // action weight for nogo
    vector[4] qv_g;  // Q value for go
    vector[4] qv_ng; // Q value for nogo
    real pGo;   // prob of go (press)

    wv_g  = initV;
    wv_ng = initV;
    qv_g  = initV;
    qv_ng = initV;

    // trial loop
    for (t in 1:nt) {
      // caluclate action weight for go and no go action
      wv_g[cue[i, t]]  = qv_g[cue[i, t]] + bias[i] + pi_[i] * val[i,t];
      wv_ng[cue[i, t]] = qv_ng[cue[i, t]];
      pGo   = inv_logit(tau[i]*(wv_g[cue[i, t]] - wv_ng[cue[i, t]]));
      {  // noise
        pGo  *= (1 - ep[i]);
        pGo   += ep[i]/2;
      }
      utility[i,t] = pGo;

      // update pavolovian state V value
      // update instrumental state action Q value
      if (choice[i, t]==2) { 
        qv_g[cue[i, t]] += alpha[i] * ( outcome[i, t] - qv_g[cue[i, t]]);
      } else { 
        qv_ng[cue[i, t]] += alpha[i] * (outcome[i, t] - qv_ng[cue[i, t]]);
      }
    } // end of trial loop
  } // end of subject loop
  
}

model {
  // hyper parameters
  mu_pr ~ std_normal();
  // Gelman recommend if the number of individual level is small, we should adopt a prior who has thinner tail.
  // like half-normal. if participants number is quite large, one could replace half normal with half-cauchy.
  sigma  ~ normal(0,3); 
  //sigma ~ cauchy(0,3);

  // individual parameters 
  alpha_raw ~ std_normal();
  ep_raw  ~ std_normal();
  bias_raw  ~ std_normal();
  pi_raw  ~ std_normal();
  tau_raw ~std_normal();
  
  for(i in 1:ns){
    for(t in 1:nt){
      choice[i,t] - 1 ~ bernoulli(utility[i,t]);
    }
  }
 
}

generated quantities {
  
  real log_lik[ns];


  { // local section, this saves time and space
    for (i in 1:ns) {
      log_lik[i] = 0;
      for (t in 1:nt) {
        log_lik[i] += bernoulli_lpmf(choice[i, t]-1 | utility[i,t]);
      } // end of i loop
   }
 }
}
