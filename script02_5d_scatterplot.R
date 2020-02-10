


plot.data = read.table('result_hc2/feature_beta.csv', sep = ',', header = T)

# select brain regions with Rsquared larger than .1.
plot.data=plot.data[plot.data$Rsquare>.1, ]

plot.data$beta1 = abs(plot.data$beta1)
plot.data$beta2 = abs(plot.data$beta2)

colnames(plot.data)[3:7] = c('age', 'age2', 'gender', 'gender_by_age', 'gender_by_age2') 




library(dplyr)

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...) {
    usr <- par("usr")
    on.exit(par(usr))
    par(usr = c(0, 1, 0, 1))
    r <- abs(cor(x, y, use = "complete.obs"))
    txt <- format(c(r, 0.123456789), digits = digits)[1]
    txt <- paste(prefix, txt, sep = "")
    if (missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
    text(0.5, 0.5, txt, cex =  cex.cor * (1 + r) / 2)
}

panel.hist <- function(x, ...) {
    usr <- par("usr")
    on.exit(par(usr))
    par(usr = c(usr[1:2], 0, 1.5) )
    h <- hist(x, plot = FALSE)
    breaks <- h$breaks
    nB <- length(breaks)
    y <- h$counts
    y <- y/max(y)
    rect(breaks[-nB], 0, breaks[-1], y, col = "white", ...)
}

#plot.new()
plot.data %>%
    select(-c(X, modality, Rsquare)) %>%
    pairs(
        upper.panel = panel.cor,
        diag.panel  = panel.hist,
        lower.panel = panel.smooth
    )
