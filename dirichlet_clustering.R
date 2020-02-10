##
# Generate some fake data with some uniform random means
##
rm(list=ls())

generateFakeData = function( num.vars=3, n=100, num.clusters=5, seed=NULL ) {
    if(is.null(seed)){
        set.seed(runif(1,0,100))
    } else {
        set.seed(seed)
    }
    data <- data.frame(matrix(NA, nrow=n, ncol=num.vars+1))
    
    mu <- NULL
    for(m in 1:num.vars){
        mu <- cbind(mu,rnorm(num.clusters, runif(1,-10,15), 5))
    }
    
    for (i in 1:n) {
        cluster <- sample(1:num.clusters, 1)
        data[i, 1] <- cluster
        for(j in 1:num.vars){
            data[i, j+1] <- rnorm(1, mu[cluster,j], 1)
        }
    }
    
    data$X1 <- factor(data$X1)
    var.names <- paste("VAR",seq(1,ncol(data)-1), sep="")
    names(data) <- c("cluster",var.names)
    
    return(data)
}

##
# Set up a procedure to calculate the cluster means using squared distance
##
dirichletClusters = function(orig.data, disp.param = NULL, max.iter = 100, tolerance = .001)
{
    n <- nrow( orig.data )
    
    data <- as.matrix( orig.data )
    pick.clusters <- rep(1, n)
    k <- 1
    
    mu <- matrix( apply(data,2,mean), nrow=1, ncol=ncol(data) )
    
    is.converged <- FALSE
    iteration <- 0
    
    ss.old <- Inf
    ss.curr <- Inf
    
    while ( !is.converged & iteration < max.iter ) { # Iterate until convergence
        iteration <- iteration + 1
        
        for( i in 1:n ) { # Iterate over each observation and measure the distance each observation' from its mean center's squared distance from its mean
            distances <- rep(NA, k)
            
            for( j in 1:k ){
                distances[j] <- sum( (data[i, ] - mu[j, ])^2 ) # Distance formula.
            }
            
            if( min(distances) > disp.param ) { # If the dispersion parameter is still less than the minimum distance then create a new cluster
                k < - k + 1
                pick.clusters[i] <- k
                mu <- rbind(mu, data[i, ])
            } else {
                pick.clusters[i] <- which(distances == min(distances))
            }
            
        }
        
        ##
        # Calculate new cluster means
        ##
        for( j in 1:k ) {
            if( length(pick.clusters == j) > 0 ) {
                mu[j, ] < - apply(subset(data,pick.clusters == j), 2, mean)
            }
        }
        
        ##
        # Test for convergence
        ##
        ss.curr <- 0
        for( i in 1:n ) {
            ss.curr <- ss.curr +
                sum( (data[i, ] - mu[pick.clusters[i], ])^2 )
        }
        ss.diff <- ss.old - ss.curr
        ss.old <- ss.curr
        if( !is.nan( ss.diff ) & ss.diff < tolerance ) {
            is.converged <- TRUE
        }
        
    }
    
    centers <- data.frame(mu)
    ret.val <- list("centers" = centers, "cluster" = factor(pick.clusters),
                    "k" = k, "iterations" = iteration)
    
    return(ret.val)
}

library(ggplot2)
create.num.vars <- 3
orig.data <- generateFakeData(create.num.vars, num.clusters=3, n=1000, seed=123)
dp.update <- dirichletClusters(orig.data[,2:create.num.vars+1], disp.param=25)
ggplot(orig.data, aes(x = VAR1, y = VAR3, color = cluster)) + geom_point()
