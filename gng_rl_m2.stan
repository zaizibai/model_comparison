// Go No Go RL
// Learning model: RW 
// Decision model: Softmax + e-greedy

data {
  int<lower=1> ns; // subject number
  int<lower=1> nt; // trial number
  int<lower=1, upper=4> cue[ns, nt]; // cue type
  int<lower=0, upper=3> choice[ns, nt]; // choice data  1 for no go response, 2 for go response
  real outcome[ns, nt]; // outcome for each action
  
}

transformed data {
  vector[4] initV;
  initV = rep_vector(0.0, 4);
}

parameters {
  vector[3] mu_pr;
  vector<lower=0>[3] sigma;
  vector[ns] alpha_raw;   // learning rate
  vector[ns] tau_raw; // inverse temperature in softmax
  vector[ns] ep_raw;

}

transformed parameters {
  vector<lower=0, upper=1>[ns] alpha; // learning rate
  vector<lower=-mu_pr[2]/sigma[2]>[ns] tau;                     // inverse temperature
  vector[ns] ep;
  matrix[ns,nt] utility;
  
  // matt-trick
  alpha = inv_logit(mu_pr[1] + sigma[1] * alpha_raw);
  tau = mu_pr[2] + sigma[2] * tau_raw;
  ep = inv_logit(mu_pr[3] + sigma[3] * ep_raw);
  
  // subject loop
  for (i in 1:ns) {
    vector[4] qv_g;  // Q value for go
    vector[4] qv_ng; // Q value for nogo
    real pGo;   // prob of go (press)
    real dw; // delta weight

    qv_g  = initV;
    qv_ng = initV;
    // trial loop
    for (t in 1:nt) {
      // caluclate action weight for go and no go action
      pGo   = inv_logit(tau[i]*(qv_g[cue[i,t]]-qv_ng[cue[i,t]]));

      {
        pGo = pGo* ( 1 - ep[i]) + ep[i]/2;
        
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
  sigma  ~ normal(0,3); 
  //sigma ~ cauchy(0,3);

  // individual parameters 
  alpha_raw ~ std_normal();
  tau_raw ~std_normal();
  ep_raw ~std_normal();

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
