library(tidyverse)
library(rstan)
library(ggplot2)
library(numDeriv) ## for computing hessian matrix
library(mltools)
library(loo)     
library(bridgesampling)
library(bmsR)   ## random effect bayesian model selection


current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)

# -------------------------------------------------------------------------------------------#

# functions for simluation and model-fitting

# -------------------------------------------------------------------------------------------#


rw_update <- function(alpha,rou,q,outcome){
  ## rescorla-wagner update rule
  # args:
  # alpha: learning rate
  # rou: sensitivity parameter

  return(alpha * ( rou * outcome - q) + q)
}

inv_logit <- function(x){
  # logistic regression(softmax/sigmoid)
  # log-sum-exp trick

  if (x >= 0) {
    z <- exp(-x)
    act  <- 1 / (1 + z)
  } else {
    z <- exp(x)
    act <- z / (1 + z)
  }

  return(act)

}


## simulate function
gng_rl_sim <- function(alpha,bias,pi,e,tau,cue,outcome){
  ## go no go RL simulator
  ## constant pavlovian bias
  ## args:
  # alpha: learning rate
  # bias: go bias
  # pi: pavlovian value weight
  # tau: temperature parameter
  # cue: cue type
  # outcome: feedback of actions

  q_go <- numeric(4)  ## instrumental q value for go response
  q_ng <- numeric(4)  ## instrumental q value for no go response
  choice <- numeric(max(dim(outcome)))

  
  for (i in sequence(max(dim(outcome)))){
    
    
    ## calculate action weight
    if(cue[i]==1 | cue[i]==3){
      val = 1 ## pavlovian value
    }
    else{
      val = -1
    }

    w_go <- q_go[cue[i]]  + pi * val + bias
    w_ng <- q_ng[cue[i]]    
    
    pGo = inv_logit(tau*(w_go-w_ng))
  
    choice[i] <- as.numeric(rbinom(1,1,pGo))+1 ## 1 for no go response, 2 for go response
    
    ## update instrumental q value

    if(choice[i]==2){
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[cue[i],choice[i],i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[cue[i],choice[i],i])
    }
    
  }

  result <- list(response = choice)
  return(result)
}

## log-likelihood function model MAP estimation
# Model 1: Full Model

llf_m1 <- function(x,choice,cue,outcome){
  ## log-likelihood function for model 1
  ## args:
  # x: parameter vector
  # choice: data
  # cue: cue type
  # outcome: feedback of actions
  
  alpha <- x[1]
  bias <- x[2]
  pi <- x[3]
  e <- x[4]
  tau <- x[5]

  q_go <- numeric(4)  ## instrumental q value for go response
  q_ng <- numeric(4)  ## instrumental q value for no go response
  llf <- numeric(max(dim(outcome)))
  for (i in sequence(max(dim(outcome)))){
    
    
    ## calculate action weight
    if(cue[i]==1 | cue[i]==3){
      val = 1
    }
    else{
      val = -1
    }

    w_go <- q_go[cue[i]]  + pi * val + bias
    w_ng <- q_ng[cue[i]]    
    
    pGo <-  inv_logit(tau*(w_go-w_ng)) * (1-e) + e/2

    llf[i] <- pGo * (choice[i] - 1) + (1 - pGo) * (2 - choice[i])
    if(llf[i]=='NaN'){
      llf[i] <- 1e-7
    }
   
    ## update instrumental q value

    if(choice[i]==2){
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[cue[i],choice[i],i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[cue[i],choice[i],i])
    }
    
  }
  
  ## sum log-likelihood and adding prior

  sum_llf <- -( sum(log(llf))) - log(dbeta(alpha,1.1,1.1))- log(dgamma(tau,2,0.3)) -
    log(dbeta(e,0.95,0.95)) - log(dbeta(bias,1.1,1.1)) - log(dbeta(pi,1.1,1.1))
  
  return(sum_llf)
}


# Model2 : model without bias and pavlovian value

llf_m2 <- function(x,choice,cue,outcome){
  ## log-likelihood function for model 2
  ## args:
  # x: parameter vector
  # choice: data
  # cue: cue type
  # outcome: feedback of actions
  
  alpha <- x[1]
  tau <- x[2]
  e <- x[3]
  
  q_go <- numeric(4)  ## instrumental q value for go response
  q_ng <- numeric(4)  ## instrumental q value for no go response
  llf <- numeric(max(dim(outcome)))

  for (i in sequence(max(dim(outcome)))){
    
    pGo <-  inv_logit(tau*(q_go[cue[i]]-q_ng[cue[i]])) * (1-e) + e/2
    
    llf[i] <- pGo * (choice[i] - 1) + (1 - pGo) * (2 - choice[i])
    if(llf[i]=='NaN'){
      llf[i] <- 1e-7
    }
    
    ## update instrumental q value

    if(choice[i]==2){
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[cue[i],choice[i],i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[cue[i],choice[i],i])
    }
    
  }
  
  ## sum log-likelihood and adding prior

  sum_llf <- - sum(log1p(llf-1)) - log(dbeta(alpha,1.1,1.1))- log(dgamma(tau,2,0.3)) -
    log(dbeta(e,0.95,0.95))
  
  return(sum_llf)
}

# -------------------------------------------------------------------------------------------#

# simulate data

# -------------------------------------------------------------------------------------------#

## sample parameter value from truncated normal distribution

set.seed(343)

alpha_true <- truncnorm::rtruncnorm(n = agent_n, a = 0, b = 1, mean = 0.48, sd = 0.2)
bias_true <- truncnorm::rtruncnorm(n = agent_n, a = 0, b = 1, mean = 0.1, sd = 0.12)
pi_true <- truncnorm::rtruncnorm(n = agent_n, a = 0, b = 1, mean = 0.1, sd = 0.2)
tau_true <- truncnorm::rtruncnorm(n = agent_n, a = 1, b = 10, mean = 4.5, sd = 2)
e_true <- runif(n=agent_n,0,0.14)

## sample outcome for each cue
## each cue has 45 trials, in total 180 trials
## cue1: go for reward
## cue2: go for avoid punishment
## cue3: no go for reward
## cue4: no go for avoid punishment

cue <- sample(rep(c(1,2,3,4),45),180)
outcome <- array(dim=c(4,2,180))
pos_outcome <- sample(c(rep(1,45*0.8),rep(0,45*0.2)),45)
zero_outcome <- sample(c(rep(0,45*0.8),rep(1,45*0.2)),45)
neg_outcome <- sample(c(rep(-1,45*0.8),rep(0,45*0.2)),45)

## framing outcome to 1,0,-1
outcome[1,1,] <- zero_outcome
outcome[1,2,] <- pos_outcome
outcome[2,1,] <- neg_outcome
outcome[2,2,] <- zero_outcome
outcome[3,1,] <- pos_outcome
outcome[3,2,] <- zero_outcome
outcome[4,1,] <- zero_outcome
outcome[4,2,] <- neg_outcome


## simulate 10 agents data

agent_n <- 10
sim_choice_rl <-  matrix(nrow = agent_n, ncol = 180) ## save simulation data

for(i in seq(agent_n)){
  ## parameter rho overlaps with inverse temperature, therefore we fix it at 1.

  sim_dat <- gng_rl_sim(alpha = alpha_true[i],bias =bias_true[i],pi = pi_true[i],e = e_true[i],
                       tau = tau_true[i],cue = cue,outcome = outcome)
  sim_choice_rl[i,] <- sim_dat$response
}

##  plot simulation data

## function for gaussian smoothing

gaussian_kernel <- function(n, sigma){
  ## n: bandwidth
  ## sigma: kernel sigma

  x <- seq(-n/2, n/2, length.out = n)
  return(exp(-x^2/(2*sigma^2)) / sqrt(2*pi*sigma^2))
}

convolution <- function(y, k){

  n <- length(y)
  m <- length(k)
  y_padded <- c(rep(0, m), y, rep(0, m))  # padding
  y_smooth <- rep(0, n)
  for(i in 1:n){
    y_smooth[i] <- sum(y_padded[i:(i+m-1)] * k)
  }
  return(y_smooth)
}


## function for plotting simulation data

plot_data <- function(data,plot_se,smooth=TRUE,linetype){
  
  ## compute SE for each cue

  se <- function(x) sd(x)/sqrt(length(x))
  choice_mu <- apply(data,2,mean)
  choice_se <- apply(data-1,2,se)

  ## smooth data
  kernel = gaussian_kernel(4,1)
  y_<- ifelse(smooth == TRUE, convolution(choice_mu,kernel), choice_mu) 

  df <- tibble(choice_mu=y_,choice_se=choice_se,t_num=1:180,cue_t = cue_t,cue=cue)
  
  df$cue <- factor(df$cue,
                   levels = c(1,2,3,4),
                   labels=c('Go_reward','Go_avoid_punishment','No_Go_reward','No_Go_avoid_punishment'))
  
  plot = ggplot()+
    geom_line(data=df,aes(x=cue_t,y=choice_mu-1,color = cue),linetype=linetype)
  
  if (plot_se==TRUE){

    ## add 0.3 SE to the mean line
    plot <- plot + geom_ribbon(data=df,aes(x=cue_t,y=choice_mu-1,color = cue,
                                           ymin = (choice_mu - 0.3*choice_se - 1), ymax = (choice_mu + 0.3*choice_se - 1),fill=cue),alpha=0.1,colour=NA)+

      
      theme(text=element_text(size=15))+
      xlab("trial")+
      ylab('go proportion')
  }
  else{

    plot <- plot+      
      theme(text=element_text(size=15))+
      xlab("trial")+
      ylab('go proportion')
  }
    
  return(plot)
}

## cue with trial number(for plotting)

cue_t <- vector(length=180) 
i_1 = 1
i_2 = 1
i_3 = 1
i_4 = 1

for (i in 1:180){

  if (cue[i]==1){
    cue_t[i] <-  i_1
    i_1 <-  i_1 + 1
  }
  else if(cue[i]==2){
    cue_t[i] <-  i_2
    i_2 <- i_2 + 1
  }
  else if(cue[i]==3){
    cue_t[i] <-  i_3
    i_3 <- i_3 + 1
  }
  else if(cue[i]==4){
    cue_t[i] <- i_4
    i_4 <- i_4 + 1
  }
}

# plot simulation data

plot_data(sim_choice_rl,plot_se = TRUE,linetype = 'solid')



# -------------------------------------------------------------------------------------------#

# Fit model with simulated data

## fit  model with RStan(MCMC)

# -------------------------------------------------------------------------------------------#



## make data list for stan fitting

val_rep <- rep(sim_dat$val,each=agent_n)
cue_fit <- rep(cue,each=agent_n)
dim(cue_fit) <- c(agent_n,180)
val_fit <- matrix(val_rep, nrow = agent_n, byrow = TRUE)
outcome_fit <- matrix(nrow=agent_n,ncol=180)

for (i in 1:agent_n){
  for(x in 1:180){
    outcome_fit[i,x] <- outcome[cue[x],sim_choice_rl[i,x],x]
  }
}

fit_list <- list(ns = 10, nt=180, cue=cue_fit,outcome=outcome_fit,choice=sim_choice_rl,val=val_fit)
fit_list2 <- list(ns = 10, nt=180, cue=cue_fit,outcome=outcome_fit,choice=sim_choice_rl)

## fit model1 full model

fit <-  stan(file = 'gng_rl.stan', data = fit_list,chains = 2,
             iter=5000,warmup = 2500,cores=2)

## fit model2 without go bias and pavlovian bias

fit2 <-  stan(file = 'gng_rl_m2.stan', data = fit_list2,chains = 2,iter=5000,warmup = 2500,cores=2)





# -------------------------------------------------------------------------------------------#

# Fit model with simulated data

## fit model with maximum a posterior estimation

# -------------------------------------------------------------------------------------------#

## fit full model

map_m1 <- matrix(nrow = agent_n,ncol = 6) ## save result

for(n in 1:agent_n){
 
  obj_fun <- 1e10 ## objective function value
  
  ## boundary for parameter searching
  lower <- c(0.001,0.001,0.001,0.001,0.001)
  upper <- c(0.999,0.999,0.999,0.15,10)

  for(i in 1:30){ ## repeat model-fitting procedure 30 times, pick the lowest negative likelihood one
    x0 <- vector(length = 5)
    
    for(x in 1:5){
      x0[x] <- runif(1,lower[x],upper[x]) ## sample initial value for parameter search
    }

    fit_result <- optim(x0,fn=llf_m1,lower = lower,upper = upper,method = 'L-BFGS-B',
                        outcome=outcome,choice=sim_choice_rl[n,],cue=cue)
    
    if(map_m1[n,6] <- obj_fun){
      map_m1[n,1:5] <- fit_result$par ## save fitted parameter
      map_m1[n,6] <- fit_result$value ## save negative log-likelihood + prior probability
      obj_fun <- fit_result$value
    }
   
  }
}

## fit model 2

map_m2 <- matrix(nrow = agent_n,ncol = 4) 

for(n in 1:agent_n){

  obj_fun <- 1e10
  lower <- c(0.001,0.001,0.001)
  upper <- c(0.999,10,0.15)
  
  for(i in 1:30){
    x0 <- vector(length = 3)
    
    for(x in 1:3){
      x0[x] <- runif(1,lower[x],upper[x])
    }
    
    fit_result2 <- optim(x0,fn=llf_m2,lower = lower,upper = upper,method = 'L-BFGS-B',
                        outcome=outcome,choice=sim_choice_rl[n,],cue=cue)
    
    if(map_m2[n,4] <- obj_fun){
      map_m2[n,1:3] <- fit_result2$par
      map_m2[n,4] <- fit_result2$value
      obj_fun <- fit_result2$value
    }
    
  }
}




# -------------------------------------------------------------------------------------------#

# Compute model-comparison index

## goodness of fit


# -------------------------------------------------------------------------------------------#

## get log-likelihood 

## divide prior probability

llf_m1 <- map_m1[,6] + 2*p_n + log(dbeta(map_m1[,1],1.1,1.1))+
  + log(dbeta(map_m1[,2],1.1,1.1)) + log(dbeta(map_m1[,3],1.1,1.1))+
  log(dbeta(map_m1[,4],.95,.95)) + log(dgamma(map_m1[,5],2,.3))

llf_m2 <- map_m2[,4] + 2*p_n + log(dbeta(map_m2[,1],1.1,1.1))+
  log(dbeta(map_m2[,3],.95,.95)) + log(dgamma(map_m2[,2],2,.3))


## posterior predictive  check
## plot two model simulation data

sim_tib <- tibble(choice_mu=apply(sim_choice_rl,2,mean),t_num=1:180,cue_t = cue_t,cue=cue)
sim_tib$cue <- factor(sim_tib$cue,
                 levels = c(1,2,3,4),
                 labels=c('Go_reward','Go_avoid_punishment','No_Go_reward','No_Go_avoid_punishment'))

# model 1

draws_m1 <- extract(fit)
pred_m1 <- apply(draws_m1$utility,c(2,3),mean)## get model prediction
plot_data(pred_m1+1,plot_se = TRUE,linetype = 'solid')

# model 2

draws_m2 <- extract(fit2)
pred_m2 <- apply(draws_m2$utility,c(2,3),mean)
plot_data(pred_m2+1,plot_se = TRUE,linetype = 'solid')

## pseudo r square
## model1(full model)

log_lf1_mu <- llf_m1
p_r2 <-  1- log_lf1_mu /180 * log(0.5)
mean(p_r2)
# 0.8143399


## model2

log_lf2_mu <- llf_m2
p_r2_2 <-  1- log_lf2_mu /180 * log(0.5)
mean(p_r2_2)
#  0.8028894

## AUC analysis

## function for computing true positive rate and false positive rate

tpr_fpr_compute <- function(pred,data){
  ## agrs:
  # pred: model predicition
  # data: 

  size <- length(pred)
  TP <-  0
  FP <-  0
  TN <-  0
  FN <-  0
  
  for(i in 1:size){
    if(pred[i]==1 & data[i]==1){
      TP <- TP +1
    }
    else if(pred[i]==1 & data[i]!=pred[i]){
      FP <- FP + 1
    }
    else if(pred[i]==0 & data[i]==0){
      TN <- TN + 1
    }
    else if(pred[i]==0 & data[i]!=pred[i]){
      FN <- FN + 1
    }
  }
  
  tpr <- TP/(TP+FN)
  fpr <- FP/(FP+TN)
  return(list(tpr=tpr,fpr=fpr))
}

## model1

## compute tpr and fpr based on different decision criterion

criterion <- seq(from=0,to=1,by=0.1)
tpr_list <- c()
fpr_list <- c()
pred_flat <- as.vector(pred_m1)
data_flat <- as.vector(fit_list$choice)-1

for(i in 1:length(criterion)){
  
  pred_cri <- c()
  
  for(t in 1:length(pred_flat)){
    
    if(pred_flat[t]>criterion[i]){
      pred_cri <- c(pred_cri,1)
    }
    else{
      pred_cri <- c(pred_cri,0)
    }
  }
  
  tpr_fpr <- tpr_fpr_compute(pred_cri,data_flat)
  tpr_list <- append(tpr_list,tpr_fpr$tpr)
  fpr_list <- append(fpr_list,tpr_fpr$fpr)
  
}

## model2

## compute tpr and fpr based on different decision criterion

tpr_list2 <- c()
fpr_list2 <- c()
pred_flat2 <- as.vector(pred_m2)

for(i in 1:length(criterion)){
  
  pred_cri <- c()
  
  for(t in 1:length(pred_flat)){
    
    if(pred_flat2[t]>criterion[i]){
      pred_cri <- c(pred_cri,1)
    }
    else{
      pred_cri <- c(pred_cri,0)
    }
  }
  
  tpr_fpr <- tpr_fpr_compute(pred_cri,data_flat)
  tpr_list2 <- append(tpr_list2,tpr_fpr$tpr)
  fpr_list2 <- append(fpr_list2,tpr_fpr$fpr)
  
}

## plot RUC
png(filename = "auc_.png", width = 800, height = 600)

plot(fpr_list,tpr_list,type = "l",col='red', main="ROC",ylab="TPR",xlab="FPR")
lines(fpr_list2,tpr_list2,col='green')
legend( 'right',legend = c("Model 1", "Model 2"), col = c("red", "green"), lty = 1)
dev.off()

## Compute AUC
auc_roc(pred_flat, data_flat)
# 0.9560354
auc_roc(pred_flat2,data_flat)
# 0.95122


# -------------------------------------------------------------------------------------------#

# Compute different model-comparison index

## cross-validation index


# -------------------------------------------------------------------------------------------#


## aic and bic

# model1

p_n <- 5
n <- 180
aic_m1 <- 2*llf_m1 + 2 * p_n
aic_m1

bic_m1 <- 2*llf_m1 + log(n)*p_n
bic_m1

# model2

p_n <- 3
aic_m2 <- 2*llf_m2 + 2*p_n
bic_m2 <- 2*llf_m2 + log(n)*p_n

## DIC:

# model1

dic_m1 <- -2*colMeans(log_lf) + apply(log_lf,2,var)

# model2

dic_m2 <- -2*colMeans(log_lf2) + apply(log_lf2,2,var)


## WAIC and Parto-Loo-CV

## Loo-CV

# model1

loo_m1 <- loo(fit)

# model2

loo_m2<- loo(fit2)

## WAIC

# model 1

waic_m1 <- waic(extract_log_lik(fit))

# model 2

waic_m2 <- waic(extract_log_lik(fit2))


## compare aic, dic, waic and loo-cv

m1_ic_tibble <- tibble(AIC=aic_m1,DIC=dic_m1,WAIC=waic_m1$pointwise[,3],
                       LOOIC=loo_m1$pointwise[,4],model='1')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "cv")
m2_ic_tibble <- tibble(AIC=aic_m2,DIC=dic_m2,WAIC=waic_m2$pointwise[,3],
                       LOOIC=loo_m2$pointwise[,4],model='2')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "cv")

ic_tibble <- bind_rows(m1_ic_tibble,m2_ic_tibble)
ic_tibble

ic_tibble %>%
  group_by(method,model) %>%
  summarise(mu=mean(cv),se=sd(cv) / sqrt(n()))%>%ggplot()+
  geom_bar(aes(x=model,y=mu,fill=model),stat = "identity",width=0.3)+
  geom_errorbar(aes(x=model, ymin=mu+se, ymax=mu-se), width = 0.1)+
  facet_wrap(~ method,ncol = 4)+
  xlab("Model")+
  ylab('Informaton criterion')+
  scale_x_discrete(labels = c("Model 1", "Model 2"))

ggsave('cv_.png',width = 10, height = 5, dpi = 300)

## using chi-square test check model difference

## aic
## calculate elpd_difference and elpd_se based on formula 24) in Vehtari, Gelman  & Gebary, 2017
sum_aic <- sum((aic_m1-aic_m2)/-2)
se_aic <- sqrt(agent_n) * sd((aic_m1-aic_m2)/-2)
sum_aic
#44.51906
se_aic
#28.21764

## dic
sum_dic <- sum((dic_m1-dic_m2)/-2)
se_dic <- sqrt(agent_n) * sd((dic_m1-dic_m2)/-2)
sum_dic
# 25.03805
se_dic
# 11.65864


# waic
loo_compare(waic_m1,waic_m2)
# elpd_diff se_diff
# model1   0.0       0.0  
# model2 -22.7      11.0

loo_compare(loo_m1,loo_m2)
# elpd_diff se_diff
# model1   0.0       0.0  
# model2 -22.7      11.0


# -------------------------------------------------------------------------------------------#

# Compute different model-comparison index

## marginal likelihood index


# -------------------------------------------------------------------------------------------#

## Laplace approximation for marginal-likelihood

## model 1

lap_lme_m1 <- vector(length = agent_n)##save result
k <- 5 ## parameter number

for(n in 1:agent_n){

  # calculate determinant of hessian matrix
  hess_det <- det(-hessian(llf_m1,map_m1[n,1:5],outcome=outcome,cue=cue,choice=sim_choice_rl[n,]))

  if (is.na(log(hess_det))==TRUE){
    lap_lme_m1[n] <- bic_m1[n]/-2  ## if log hessian matrix determinant is NaN, replace it with BIC
  }
  else{
    lap_lme_m1[n] <- -map_m1[n,6] +  k/2 * log(2*pi) - log(hess_det)/2
    
  }
}


## model 2

lap_lme_m2 <- vector(length = agent_n)
k <- 3

for(n in 1:agent_n){

  hess_det <- det(-hessian(llf_m2,map_m2[n,1:3],outcome=outcome,cue=cue,choice=sim_choice_rl[n,]))

  if (is.na(log(hess_det))==TRUE){
    lap_lme_m2[n] <- bic_m2[n]/-2
  }
  else{
    lap_lme_m2[n] <- -map_m2[n,4] +  k/2 * log(2*pi) - log(hess_det)/2
  }

}

## bridge-sampling  marginal-likelihood

# model1

model1_n <- stan('gng_rl.stan',data=fit_list,iter = 1)  ## prevent bridge sampling from crashing in windows
bs_lme_m1 <- bridge_sampler(fit,model1_n, silent = TRUE)

# model2

model2_n <- stan('gng_rl_m2.stan',data=fit_list,iter = 1)
bs_lme_m2 <- bridge_sampler(fit2,model2_n,silent=TRUE)


## compare marginal likelihood index

m1_lme_tibble <- tibble(bic=sum(bic_m1)/-2,bs=bs_lme_m1$logml,
                        lap=sum(lap_lme_m1),model='1')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "lme")
m2_lme_tibble <- tibble(bic=sum(bic_m2)/-2,bs=bs_lme_m2$logml,
                        lap=sum(lap_lme_m2),model='2')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "lme")

lme_tibble <- bind_rows(m1_lme_tibble,m2_lme_tibble)%>%
  mutate(
    method=str_replace(method,'bic','BIC'),
    method=str_replace(method,'bs','Bridge-Sampling'),
    method=str_replace(method,'lap','Laplace-Approximation'))

ggplot()+geom_bar(data=lme_tibble,aes(x=model,y=lme,fill=model),stat = "identity",width=0.3, fun.y = "mean")+
  facet_wrap(~ method)+
  xlab("Model")+
  ylab('Log marginal likelihood')+
  scale_x_discrete(labels = c("Model 1", "Model 2"))
  

ggsave('lme_.png',width = 10, height = 5, dpi = 300)



## bayes factor analysis

## Savage-Dickey Ratio

## get posterior samples of group-level mu bias and pi parameter

bias_m_sam <- draws_m1$mu_pr[,3]
pi_m_sam <- draws_m1$mu_pr[,4]


## Using KDE to compute probability that parameter = 0

## group-level mean parameters
bias_m_kde <- density(bias_m_sam)
bias_m_post_0 <- approx(bias_m_kde$x, bias_m_kde$y, xout = 0)$y
pi_m_kde <- density(pi_m_sam)
pi_m_post_0 <- approx(pi_m_kde$x,pi_kde$y,xout = 0)$y


## compute Savage-Dickey Ratio version bayes factor

bf_sd <- bias_m_post_0 * pi_m_post_0 / (dnorm(0)*dnorm(0))
log(bf_sd)
# 2.238351

## BIC version bayes factor

log(exp(sum(bic_m1-bic_m2)/-2))
# 12.58949

##  bridge sampling version bayes factor

log(exp((bs_lme_m1$logml-bs_lme_m2$logml)))
# 39.92257

## laplace approximation version bayes factor
log(exp(sum(lap_lme_m1-lap_lme_m2)))
# 50.62551


## random effect bayesian model selection

## input of random effect model selction is N X M matrix
## N is the participant number
## M is the model number

com_matrix <- matrix(nrow=10,ncol = 2)
com_matrix[,1] <- lap_lme_m1
com_matrix[,2] <- lap_lme_m2

VB_bms(com_matrix)

# $alpha
# [1] 7.258061 4.741939
# 
# $r
# [1] 0.6048384 0.3951616
# 
# $xp
# [1] 0.775073 0.224927
# 
# $bor
# [1] 0.6806362
# 
# $pxp
# [1] 0.5878483 0.4121517

## save result
save.image('model_comparison.RData')
