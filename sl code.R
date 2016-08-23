# R code that creates SL analysis for R meetup group
## Created: 4/11/2016
## Modified:
## What is done:
## TO DO:
## ** Notes **


rhc <- read.csv("E:/KSU stuff/Course materials/Survival course/Resources/Code and data sets/Right Heart Catheterization Dataset/rhc2.csv")

# Subset data to include only ARF subjects
arf <- subset(rhc, cat1=='ARF') 
# delete unused variables
arf$cat1 <- arf$cat2 <- arf$cancer <- arf$sadmdte  <- arf$dschdte	<- 
  arf$dthdte	<- arf$lstctdte <- arf$t3d30 <-
  arf$death <- arf$surv2md1  <- arf$ortho <- arf$adld3p <- arf$urin1 <- 
  arf$strat <- arf$income <- arf$ninsclas <- NULL
rm(rhc) # remove rhc2 dataset

library(rms) # Load RMS package
# Descriptives
describe(arf,digits=2)

save(arf, file ="C:/Users/sfgre/Documents/Targeted Learning Presentation/arf.RData")


#### Prepare data ####
load("C:/Users/sfgre/Documents/Targeted Learning Presentation/arf.RData")

# Impute missing X values #
library("VIM")

# Scale cont vars #
library(arm)
cont <- c("age","edu","das2d3pc","aps1","scoma1","meanbp1","wblc1","hrt1",
          "resp1","temp1","pafi1","alb1","hema1","bili1","crea1","sod1",
          "pot1","paco21","ph1","wtkilo1")
arf[,cont] <- data.frame(apply(arf[cont], 2, function(x)
  {x <- rescale(x, "full")})); rm(cont) # standardizes by centering and 
                                        # dividing by 2 sd

# Create dummy vars #
arf$rhc <- ifelse(arf$swang1=="RHC",1,0)
arf$white <- ifelse(arf$race=="white",1,0)  
arf$swang1 <- arf$race <- NULL

arf$death <- arf$dth30; arf$dth30 <- NULL
arf$id <- arf$ptid; arf$ptid <- NULL

#### Prepare Super Learner ####
# Recomended packages, if needed
# install.packages(  c("glmnet","randomForest","class","gam","gbm","nnet",
#                    "polspline","MASS","e1071","stepPlr","arm","party",
#                    "spls","LogicReg","nnls","multicore","SIS","BayesTree",
#                    "ipred","mlbench","rpart","caret","mda","earth"))

library(SuperLearner)
listWrappers() # Look at SL prediction algorithm wrappers
SL.glmnet      # Look at the elastic net wrapper

# Specify new SL prediction algorithm wrapper for Ridge regression #
SL.glmnet.0 <- function(..., alpha = 0){
  SL.glmnet(..., alpha = alpha) 
  } 

# Specify the SL library with prediction algorithms to be used #

SL.library <- c("SL.glm","SL.bayesglm","SL.earth","SL.gam","SL.glmnet",
                "SL.glmnet.0","SL.knn","SL.step","SL.nnet")

#### Run SuperLearner 1st prediction model ####

library(parallel)
detectCores(all.tests = FALSE, logical = TRUE)

system.time({
  pm1 <- CV.SuperLearner(Y=arf$death, 
                         X=arf[1:45], 
                         V=10, family=binomial(),
                         SL.library=SL.library, 
                         method="method.NNLS",
                         verbose = TRUE,
                         control = list(saveFitLibrary = TRUE),
                         cvControl = list(V=10), saveAll = TRUE,
                         parallel = 'multicore')
  
})[[3]] # Obtain computation time: 66.689 min.

save(pm1, file ="C:/Users/sfgre/Documents/pm1.RData")

# SuperLearner prediction model
load("C:/Users/sfgre/Documents/pm1.RData")

# Print CV Mean Squared Error 
summary(pm1)

# Plot CV Mean Squared Error
plot(pm1, packag ="ggplot2")

# Best algorithm for each V-fold
pm1$whichDiscreteSL

# Average of SL alpha weights
colMeans(pm1$coef)

# Plot the ROC curves for each algorithm
library(pROC)
predictions <- as.data.frame(cbind(pm1$library.predict))
SL.predict <- pm1$SL.predict; Y <- pm1$Y
predictions <- cbind(predictions, SL.predict, Y); rm(SL.predict,Y)

colors <- palette(rainbow(9))
plot.roc(predictions$Y, predictions$SL.predict, col="black", lwd=2)
for(i in 1:length(predictions[1:9])) {
  plot.roc(predictions$Y, predictions[[i]], col=colors[i], lwd=1, add=TRUE)
  }
plot.roc(predictions$Y, predictions$SL.predict, col="black", lwd=2, add=TRUE)

freqta <- apply(predictions[1:10], 2, function(x){
  a <- roc(pm1$Y, x, ci=TRUE, of="auc")
  auc <- round(auc(a),3)
  lci <- round(ci(a)[[1]],3)
  uci <- round(ci(a)[[3]],3)
  row <- rbind(paste(auc," (",lci,"-",uci,")", sep = "", collapse = NULL))
  })
table <- as.data.frame(freqta); rm(freqta)

#### Run SuperLearner 2st prediction model ####

colnames(arf)
# Specify the SL screening algorithm wrapper to exclude hx vars #
screen.nohx <- function(...){
  return(c(rep(FALSE,12), rep(TRUE,33)))
  }

# Specify the SL library with both prediction & screening algorithms #
SL.library <- list(c("SL.gam","All","screen.nohx","screen.glmnet"))

#### Run SuperLearner w method.NNLS ####

system.time({
pm2 <- CV.SuperLearner(Y=arf$death, 
                       X=arf[1:45], 
                       V=10, family=binomial(),
                       SL.library, 
                       method="method.NNLS",
                       verbose = TRUE,
                       control = list(saveFitLibrary = TRUE),
                       cvControl = list(V=10), saveAll = TRUE,
                       parallel = 'multicore')

})[[3]] # Obtain computation time 3.627 min.

plot(pm2, packag ="ggplot2")
summary(pm2)

# Examine vars in best algorithm
out = c()
for(j in 1:10){
  out = c(out,names(pm2[["AllSL"]][[j]][["fitLibrary"]][[3]][[1]][["coefficients"]]))
  }
table(out); rm(out,j) 

