library(R.matlab)
library(ggplot2)
library(dplyr)
library(DEoptim)
library(rstan)
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


## loss function for MAP estimation
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
  llf <- numeric(length(choice))
  for (i in 1:length(choice)){
    
    
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
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[i])
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
  llf <- numeric(length(choice))
  
  for (i in 1:length(choice)){
    
    pGo <-  inv_logit(tau*(q_go[cue[i]]-q_ng[cue[i]])) * (1-e) + e/2
    
    llf[i] <- pGo * (choice[i] - 1) + (1 - pGo) * (2 - choice[i])
    if(llf[i]=='NaN'){
      llf[i] <- 1e-7
    }
    
    ## update instrumental q value
    
    if(choice[i]==2){
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[i])
    }
  }
  
  ## sum log-likelihood and adding prior
  
  sum_llf <- - sum(log1p(llf)) - log(dbeta(alpha,1.1,1.1))- log(dgamma(tau,2,0.3)) -
    log(dbeta(e,0.95,0.95))
  
  return(sum_llf)
}

# Model 3: Without go bias parameter

llf_m3 <- function(x,choice,cue,outcome){
  ## log-likelihood function for model 3
  ## args:
  # x: parameter vector
  # choice: data
  # cue: cue type
  # outcome: feedback of actions
  
  alpha <- x[1]
  pi <- x[2]
  e <- x[3]
  tau <- x[4]
  
  q_go <- numeric(4)  ## instrumental q value for go response
  q_ng <- numeric(4)  ## instrumental q value for no go response
  llf <- numeric(length(choice))
  for (i in 1:length(choice)){
    
    
    ## calculate action weight
    if(cue[i]==1 | cue[i]==3){
      val = 1
    }
    else{
      val = -1
    }
    
    w_go <- q_go[cue[i]]  + pi * val 
    w_ng <- q_ng[cue[i]]    
    
    pGo <-  inv_logit(tau*(w_go-w_ng)) * (1-e) + e/2
    
    llf[i] <- pGo * (choice[i] - 1) + (1 - pGo) * (2 - choice[i])
    if(llf[i]=='NaN'){
      llf[i] <- 1e-7
    }
    
    ## update instrumental q value
    
    if(choice[i]==2){
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[i])
    }
    
  }
  
  ## sum log-likelihood and adding prior
  
  sum_llf <- -( sum(log(llf))) - log(dbeta(alpha,1.1,1.1))- log(dgamma(tau,2,0.3)) -
    log(dbeta(e,0.95,0.95))  - log(dbeta(pi,1.1,1.1))
  
  return(sum_llf)
}

# Model 4: Without pavlovian bias parameter

llf_m4 <- function(x,choice,cue,outcome){
  ## log-likelihood function for model 4
  ## args:
  # x: parameter vector
  # choice: data
  # cue: cue type
  # outcome: feedback of actions
  
  alpha <- x[1]
  bias <- x[2]
  e <- x[3]
  tau <- x[4]
  
  q_go <- numeric(4)  ## instrumental q value for go response
  q_ng <- numeric(4)  ## instrumental q value for no go response
  llf <- numeric(length(choice))
  for (i in 1:length(choice)){
    
    
    ## calculate action weight
    if(cue[i]==1 | cue[i]==3){
      val = 1
    }
    else{
      val = -1
    }
    
    w_go <- q_go[cue[i]]  + bias
    w_ng <- q_ng[cue[i]]    
    
    pGo <-  inv_logit(tau*(w_go-w_ng)) * (1-e) + e/2
    
    llf[i] <- pGo * (choice[i] - 1) + (1 - pGo) * (2 - choice[i])
    if(llf[i]=='NaN'){
      llf[i] <- 1e-7
    }
    
    ## update instrumental q value
    
    if(choice[i]==2){
      q_go[cue[i]] <- rw_update(alpha,1,q_go[cue[i]],outcome[i])
    }
    else if(choice[i]==1){
      q_ng[cue[i]] <- rw_update(alpha,1,q_ng[cue[i]],outcome[i])
    }
    
  }
  
  ## sum log-likelihood and adding prior
  
  sum_llf <- -( sum(log(llf))) - log(dbeta(alpha,1.1,1.1))- log(dgamma(tau,2,0.3)) -
    log(dbeta(e,0.95,0.95)) - log(dbeta(bias,1.1,1.1))
  
  return(sum_llf)
}

# -------------------------------------------------------------------------------------------#

## Load and clean task data 

# -------------------------------------------------------------------------------------------#

#all PIT participant IDs
studyIDAll <-  c(1,2,3,4,5,7,8,9,10,12,13,14,16:27,32,33,36:40,42:44,46:49,51:56,58,131,147:150,154,156,159:167)
allPIT <- data.frame()
agent_n <- length(studyIDAll)

#loop through and create allPIT data that compiles data from each subj
current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
dataDir <- file.path(current_working_dir,'task_data//')


for(subj in 1:length(studyIDAll)){
  data <- readMat(paste0(dataDir,studyIDAll[subj],
                         "_TaskDataLearning_Session1.mat"))
  data <- as.data.frame(data)
  data <- cbind(subjID = studyIDAll[subj], data) 
  allPIT <- rbind(allPIT,data)
}

#rename columns 
## cue: 1 (go for reward), 2 (go for avoid loss), 3(no go for reward),
## 4 (no go for avoid loss)

colnames(allPIT) <-c("subjID","trialNum","trialType","timeCue","response1Cue",
                     "response2Cue","time2target","timeTarget","keyResp",
                     "keyTime","RT","Response","corrResp","foil",
                     "targetDisplayTime","timeOutcome","ITI","Won")

## make list for stan model fitting
real_dat_list <- list()
subj_num <- length(unique(allPIT$subjID))
trial_num <- max(allPIT$trialNum)
choice <- matrix(allPIT$Response,subj_num,trial_num) + 1
choice[choice %in% 6] <- 1
cue <- matrix(allPIT$trialType ,subj_num,trial_num)
outcome <- matrix(allPIT$Won, subj_num, trial_num)
val <- cue
val[val %in% c(1,3)] <- 1
val[val %in% c(2,4)] <- -1
## cue with trial number(for plotting)
cue_comb <- matrix(0,subj_num,trial_num)

for (sub in 1:subj_num){
  cue_t <- vector(length=trial_num) 
  i_1 = 1
  i_2 = 1
  i_3 = 1
  i_4 = 1
  
  for (i in 1:180){
    
    if (cue[sub,i]==1){
      cue_t[i] <-  i_1
      i_1 <-  i_1 + 1
    }
    else if(cue[sub,i]==2){
      cue_t[i] <-  i_2
      i_2 <- i_2 + 1
    }
    else if(cue[sub,i]==3){
      cue_t[i] <-  i_3
      i_3 <- i_3 + 1
    }
    else if(cue[sub,i]==4){
      cue_t[i] <- i_4
      i_4 <- i_4 + 1
    }
  }
  print(sub)
  cue_comb[sub,] <- cue_t
  
}

## make list for stan model fitting
  
fit_list <- list(ns = agent_n, nt=180, cue=cue,outcome=outcome,choice=choice,val=val)
fit_list2 <- list(ns = agent_n, nt=180, cue=cue,outcome=outcome,choice=choice)

# -------------------------------------------------------------------------------------------#

  
## Fit models with Hierarchical-model with Rstan

# -------------------------------------------------------------------------------------------#

  
## fit model1 full model

fit_real<-  stan(file = 'gng_rl.stan', data = fit_list,chains = 2,
             iter=5000,warmup = 2500,cores=2)

## fit model2 without go bias and pavlovian bias

fit2_real <-  stan(file = 'gng_rl_m2.stan', data = fit_list2,chains = 2,iter=5000,warmup = 2500,cores=2)

## fit model3 without go bias 
  
fit3_real <-  stan(file = 'gng_rl_m3.stan', data = fit_list,chains = 2,iter=5000,warmup = 2500,cores=2)

## fit model3 without pavlovian bias 
  
fit4_real <-  stan(file = 'gng_rl_m4.stan', data = fit_list2,chains = 2,iter=5000,warmup = 2500,cores=2)


# -------------------------------------------------------------------------------------------#

## Fit model with MAP estimation

# -------------------------------------------------------------------------------------------#

## fit full model


map_m1 <- matrix(nrow = agent_n,ncol = 6) ## save result
for(n in 1:agent_n){
  
  ## boundary for parameter searching
  lower <- c(0.001,-1,-1,0.001,0.001)
  upper <- c(0.999,1,1,0.999,30)
  
 
  fit_result <- DEoptim::DEoptim(fn=llf_m1,lower = lower,upper = upper,
                      outcome=outcome[n,],choice=choice[n,],cue=cue[n,])
  
  map_m1[n,1:5] <- fit_result$optim$bestmem ## save fitted parameter
  map_m1[n,6] <- fit_result$optim$bestval ## save negative log-likelihood + prior probability
}


## fit model 2

map_m2 <- matrix(nrow = agent_n,ncol = 4) 

for(n in 1:agent_n){
  
  lower <- c(0.001,0.001,0.001)
  upper <- c(0.999,10,0.15)
  fit_result2 <- DEoptim::DEoptim(fn=llf_m2,lower = lower,upper = upper,
                                  outcome=outcome[n,],choice=choice[n,],cue=cue[n,])
      
  map_m2[n,1:3] <- fit_result2$optim$bestmem
  map_m2[n,4] <- fit_result2$optim$bestval
    

}

## fit model3

map_m3 <- matrix(nrow = agent_n,ncol = 5) ## save result

for(n in 1:agent_n){
  
  ## boundary for parameter searching
  lower <- c(0.001,0.001,0.001,0.001)
  upper <- c(0.999,0.999,0.15,10)
  
  fit_result3 <- DEoptim::DEoptim(fn=llf_m3,lower = lower,upper = upper,
                                  outcome=outcome[n,],choice=choice[n,],cue=cue[n,])
     
  map_m3[n,1:4] <- fit_resul3t$optim$bestmem
  map_m3[n,5] <- fit_result3$optim$bestval ## save negative log-likelihood + prior probability
}

## fit model4

map_m4 <- matrix(nrow = agent_n,ncol = 5) ## save result

for(n in 1:agent_n){
  
  obj_fun <- 1e10 ## objective function value
  
  ## boundary for parameter searching
  lower <- c(0.001,0.001,0.001,0.001)
  upper <- c(0.999,0.999,0.15,10)
  
  fit_result4 <- DEoptim::DEoptim(fn=llf_m4,lower = lower,upper = upper,
                                  outcome=outcome[n,],choice=choice[n,],cue=cue[n,])
    
  map_m4[n,1:4] <- fit_result4$optim$bestmem
  map_m4[n,5] <- fit_result4$optim$bestval
}

# -------------------------------------------------------------------------------------------#

# Compute model-comparison index

## goodness of fit


# -------------------------------------------------------------------------------------------#

## get log-likelihood for MAP estimation

## divide prior probability

llf_m1_map <- map_m1[,6] + 2*p_n + log(dbeta(map_m1[,1],1.1,1.1))+
  + log(dbeta(map_m1[,2],1.1,1.1)) + log(dbeta(map_m1[,3],1.1,1.1))+
  log(dbeta(map_m1[,4],.95,.95)) + log(dgamma(map_m1[,5],2,.3))

llf_m2_map <- map_m2[,4] + 2*p_n + log(dbeta(map_m2[,1],1.1,1.1))+
  log(dbeta(map_m2[,3],.95,.95)) + log(dgamma(map_m2[,2],2,.3))

llf_m3_map <- map_m3[,5] + 2*p_n + log(dbeta(map_m3[,1],1.1,1.1))+
  + log(dbeta(map_m3[,2],1.1,1.1)) +
  log(dbeta(map_m3[,3],.95,.95)) + log(dgamma(map_m3[,4],2,.3))

llf_m4_map <-map_m4[,5] + 2*p_n + log(dbeta(map_m4[,1],1.1,1.1))+
  + log(dbeta(map_m4[,2],1.1,1.1)) +
  log(dbeta(map_m4[,3],.95,.95)) + log(dgamma(map_m4[,4],2,.3))
map_m4

## posterior predictive  check
## plot two model simulation data

sim_tib <- tibble(choice_mu=apply(sim_choice_rl,2,mean),t_num=1:180,cue_t = cue_t,cue=cue)
sim_tib$cue <- factor(sim_tib$cue,
                      levels = c(1,2,3,4),
                      labels=c('Go_reward','Go_avoid_punishment','No_Go_reward','No_Go_avoid_punishment'))

# model 1

draws_m1 <- extract(fit)
log_lf1_stan <- draws_m1$log_lik
pred_m1 <- apply(draws_m1$utility,c(2,3),mean)## get model prediction
plot_data(pred_m1+1,plot_se = TRUE,linetype = 'solid')

# model 2

draws_m2 <- extract(fit2)
log_lf2_stan <- draws_m2$log_lik
pred_m2 <- apply(draws_m2$utility,c(2,3),mean)
plot_data(pred_m2+1,plot_se = TRUE,linetype = 'solid')

## pseudo r square
## model1(full model)

log_lf1_mu <- llf_m1_map
p_r2 <-  1+ log_lf1_mu /(180 * log(0.5))
mean(p_r2)
# 0.5820934


## model2

log_lf2_mu <- llf_m2_map
p_r2_2 <-  1+ (log_lf2_mu /(180 * log(0.5)))
mean(p_r2_2)
# 0.5154812

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
tib <- tibble(FPR=c(fpr_list,fpr_list2),TPR=c(tpr_list,tpr_list2),Model=as.character(c(rep(1,length(fpr_list)),rep(2,length(fpr_list2)))))
ggplot(data=tib)+geom_line(aes(x=FPR,y=TPR,color=Model),size=1.2)+theme_classic() + theme(plot.title = element_text(hjust = 0.5,size = 20),
                                                                                          axis.title.x = element_text(size = 16),  
                                                                                          axis.title.y = element_text(size = 16),
                                                                                          axis.text.x = element_text(size = 15),
                                                                                          axis.text.y = element_text(size = 15),
                                                                                          legend.position = c(0.05, 0.9),
                                                                                          legend.background = element_rect(color = "black"),
                                                                                          legend.text = element_text(size = 17),
                                                                                          legend.margin = margin(t = 5, l = 5, r = 5, b = 5),
                                                                                          legend.key = element_rect(color = NA, fill = NA)
)

ggsave('auc.png',dpi=200,width=10,height = 7)


## Compute AUC
auc_roc(pred_flat, data_flat)
# 0.948447
auc_roc(pred_flat2,data_flat)
# 0.9457004



## compute likelihood-ratio test

ratio <- sum(2*(-llf_m1_map + llf_m2_map) )
p_value <- pchisq(ratio, agent_n * 2, lower.tail = FALSE)
p_value
#  1.458517e-198


# -------------------------------------------------------------------------------------------#

# Compute different model-comparison index

## cross-validation index


# -------------------------------------------------------------------------------------------#


## aic and bic

# model1

p_n <- 5
n <- 180
aic_m1 <- 2*llf_m1_map + 2 * p_n
aic_m1

bic_m1 <- 2*llf_m1_map + log(n)*p_n
bic_m1

# model2

p_n <- 3
aic_m2 <- 2*llf_m2_map+ 2*p_n
bic_m2 <- 2*llf_m2_map + log(n)*p_n

## model3

p_n <- 4
n <- 180
aic_m3 <- 2*llf_m3_map + 2 * p_n
aic_m3

bic_m3 <- 2*llf_m3_map + log(n)*p_n
bic_m3

## model4

p_n <- 4
n <- 180
aic_m4 <- 2*llf_m4_map + 2 * p_n
aic_m4

bic_m4 <- 2*llf_m4_map + log(n)*p_n
bic_m4


## DIC:

# model1

dic_m1 <- -2*colMeans(log_lf1_stan) + apply(log_lf1_stan,2,var)

# model2

dic_m2 <- -2*colMeans(log_lf2_stan) + apply(log_lf2_stan,2,var)
dic_m2

## WAIC and Parto-Loo-CV

## Loo-CV

loo_m1 <- loo(fit)
loo_m2<- loo(fit2)
loo_m3<- loo(fit3)
loo_m4<- loo(fit4)
  
## WAIC

waic_m1 <- waic(extract_log_lik(fit))
waic_m2 <- waic(extract_log_lik(fit2))
waic_m3 <- waic(extract_log_lik(fit3))
waic_m4 <- waic(extract_log_lik(fit4))


## compare aic, dic, waic and loo-cv

m1_ic_tibble <- tibble(AIC=aic_m1,DIC=dic_m1,WAIC=waic_m1$pointwise[,3],
                       LOOIC=loo_m1$pointwise[,4],model='1')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "cv")
m2_ic_tibble <- tibble(AIC=aic_m2,DIC=dic_m2,WAIC=waic_m2$pointwise[,3],
                       LOOIC=loo_m2$pointwise[,4],model='2')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "cv")

ic_tibble <- bind_rows(m1_ic_tibble,m2_ic_tibble)

offset_ic <- min(ic_tibble$cv)
ic_tibble %>%
  group_by(method,model) %>%
  summarise(mu=mean(cv),se=sd(cv) / sqrt(n()))%>%ggplot(aes(x=method,y=mu-offset_ic,fill=model,ymin=mu+se-offset_ic, ymax=mu-se-offset_ic))+
  geom_bar(position = position_dodge(.35),stat = "identity",width=0.3)+
  geom_errorbar(position = position_dodge(.35), width = 0.1)+
  xlab(' ')+
  ylab('Informaton criterion')+
  scale_x_discrete(labels = c("AIC", "DIC","WAIC",'LOOIC'))+
  theme_classic()+
  theme(axis.title = element_text(size = 20),
        axis.text = element_text(size = 20, color = "black"),
        axis.text.x = element_text(margin = margin(t = 10)),
        axis.text.y = element_text(size = 15),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.ticks.x = element_blank(),
        legend.position = c(0.05, 0.8),
        legend.background = element_rect(color = "black"),
        legend.text = element_text(size = 15),
        legend.margin = margin(t = 5, l = 5, r = 5, b = 5),
        legend.key = element_rect(color = NA, fill = NA))+
  scale_y_continuous(limits = c(0, 50 ), 

                     labels = seq(150, 200, by = 10))



ggsave('cv_.png',width = 10, height = 5, dpi = 300)

## using chi-square test check model difference

## aic
## calculate elpd_difference and elpd_se based on formula 24) in Vehtari, Gelman  & Gebary, 2017
sum_aic <- sum((aic_m1-aic_m2)/-2)
se_aic <- sqrt(agent_n) * sd((aic_m1-aic_m2)/-2)
sum_aic
#378.6581
se_aic
#63.8442

## dic
sum_dic <- sum((dic_m1-dic_m2)/-2)
se_dic <- sqrt(agent_n) * sd((dic_m1-dic_m2)/-2)
sum_dic
#  126.5891
se_dic * 1.96
# 49.35669


# waic
waic_diff <- loo_compare(waic_m1,waic_m2)
# elpd_diff se_diff
# model1    0.0       0.0 
# model2 -124.7      25.0 
waic_diff[,'se_diff']['model2'] * 1.96

loo_diff<- loo_compare(loo_m1,loo_m2)
# elpd_diff se_diff
# model1    0.0       0.0 
# model2 -126.0      24.9
loo_diff[,'se_diff']['model2'] * 1.96

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

## model 3

lap_lme_m3 <- vector(length = agent_n)##save result
k <- 4 ## parameter number

for(n in 1:agent_n){
  
  # calculate determinant of hessian matrix
  hess_det <- det(-hessian(llf_m3,map_m3[n,1:4],outcome=outcome,cue=cue,choice=sim_choice_rl[n,]))
  
  if (is.na(log(hess_det))==TRUE){
    lap_lme_m3[n] <- bic_m3[n]/-2  ## if log hessian matrix determinant is NaN, replace it with BIC
  }
  else{
    lap_lme_m3[n] <- -map_m3[n,6] +  k/2 * log(2*pi) - log(hess_det)/2
    
  }
}

## model 4

lap_lme_m4 <- vector(length = agent_n)##save result
k <- 4 ## parameter number

for(n in 1:agent_n){
  
  # calculate determinant of hessian matrix
  hess_det <- det(-hessian(llf_m4,map_m4[n,1:4],outcome=outcome,cue=cue,choice=sim_choice_rl[n,]))
  
  if (is.na(log(hess_det))==TRUE){
    lap_lme_m4[n] <- bic_m4[n]/-2  ## if log hessian matrix determinant is NaN, replace it with BIC
  }
  else{
    lap_lme_m4[n] <- -map_m4[n,6] +  k/2 * log(2*pi) - log(hess_det)/2
    
  }
}


## bridge-sampling  marginal-likelihood

# model1

model1_n <- stan('gng_rl.stan',data=fit_list,iter = 1)  ## prevent bridge sampling from crashing in windows
bs_lme_m1 <- bridge_sampler(fit_real,model1_n, silent = TRUE)

# model2

model2_n <- stan('gng_rl_m2.stan',data=fit_list2,iter = 1)
bs_lme_m2 <- bridge_sampler(fit2_real,model2_n,silent=TRUE)

# model3

model3_n <- stan('gng_rl_m3.stan',data=fit_list,iter = 1)  ## prevent bridge sampling from crashing in windows
bs_lme_m3 <- bridge_sampler(fit3_real,model3_n, silent = TRUE)

# model4

model4_n <- stan('gng_rl_m4.stan',data=fit_list2,iter = 1)  ## prevent bridge sampling from crashing in windows
bs_lme_m4 <- bridge_sampler(fit4_real,model4_n, silent = TRUE)


## compare marginal likelihood index

m1_lme_tibble <- tibble(bic=sum(bic_m1)/-2,bs=bs_lme_m1$logml,
                        lap=sum(lap_lme_m1),model='1')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "lme")
m2_lme_tibble <- tibble(bic=sum(bic_m2)/-2,bs=bs_lme_m2$logml,
                        lap=sum(lap_lme_m2),model='2')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "lme")
m3_lme_tibble <- tibble(bic=sum(bic_m3)/-2,bs=bs_lme_m3$logml,
                        lap=sum(lap_lme_m3),model='3')%>%
  pivot_longer(cols = -c(model),names_to = 'method',values_to = "lme")
m4_lme_tibble <- tibble(bic=sum(bic_m4)/-2,bs=bs_lme_m4$logml,
                      lap=sum(lap_lme_m4),model='4')%>%
pivot_longer(cols = -c(model),names_to = 'method',values_to = "lme")


lme_tibble <- bind_rows(m1_lme_tibble,m2_lme_tibble,m3_lme_tibble,m4_lme_tibble)%>%
  mutate(
    method=str_replace(method,'bic','BIC'),
    method=str_replace(method,'bs','Bridge-Sampling'),
    method=str_replace(method,'lap','Laplace-Approximation'))


offset_lme <- 5000
ggplot(data=lme_tibble,aes(x=method,y=lme+offset_lme,fill=model))+
  geom_col(position = position_dodge(.35),width=0.3)+
  xlab(" ")+
  ylab('Log marginal likelihood')+
  scale_x_discrete(labels = c("BIC", "Laplace",'Bridge-Sampling'))+
  theme_classic()+
  theme(axis.title = element_text(size = 20),
        axis.text = element_text(size = 20, color = "black"),
        axis.text.x = element_text(margin = margin(t = 10)),
        axis.text.y = element_text(size = 15),
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.ticks.x = element_blank(),
        legend.position = c(0.05, 0.8),
        legend.background = element_rect(color = "black"),
        legend.text = element_text(size = 15),
        legend.margin = margin(t = 5, l = 5, r = 5, b = 5),
        legend.key = element_rect(color = NA, fill = NA))+
  scale_y_continuous(limits = c(-8000 + offset_lme, -5000 + offset_lme), 
                      breaks = seq(-8000 + offset_lme, -5000 + offset_lme, by = 500),
                      labels = seq(-8000, -5000, by = 500)) 
lme_tibble


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

bf_sd <-  pi_m_post_0 *bias_m_post_0 / (dnorm(0)*dnorm(0))
log(bf_sd)


## BIC version bayes factor

log(exp(sum(bic_m1-bic_m2)/-2))
# 187.0807

##  bridge sampling version bayes factor

log(exp((bs_lme_m1$logml-bs_lme_m2$logml)))
# 230.691

## laplace approximation version bayes factor
log(exp(sum(lap_lme_m1-lap_lme_m2)))
# 312.0116

## model average using BIC

comb_bic <- cbind(bic_m1,bic_m2,bic_m3,bic_m4)
bic_extract <- comb_bic - apply(comb_bic,MARGIN = 1,FUN = min)
bic_weight <- exp(-0.5*bic_extract)/ (apply(exp(bic_extract*-0.5),MARGIN = 1, FUN = sum))
bic_bias <- bic_weight[,1]+bic_weight[,4]
bic_without_bias <- bic_weight[,2]+bic_weight[,3]
bf_bias <- bic_bias/bic_without_bias
log(sum(bf_bias))
## 8.290493
log(sum(bic_weight[,1]/bic_weight[,3]))
## 8.607969

bic_pi <- bic_weight[,1]+bic_weight[,3]
bic_without_pi <- bic_weight[,2]+bic_weight[,4]
bf_pi <- bic_pi/bic_without_pi
log(sum(bf_pi))
## 25.93274
log(sum(bic_weight[,1]/bic_weight[,4]))
## 27.85304




