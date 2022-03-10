p.values = read.csv()

p <- c(0.015, 0.036, 0.048, 0.052, 0.023 )
p.bon = p.adjust(p, "bonferroni")
p.adjust(p, "holm")
p.adjust(p, "hochberg")
p.adjust(p, "hommel")



x = rnorm(10)
y = .1*x+rnorm(mean = 10, sd = 2)
plot(x,y)

data = data.frame(x=x, y=y)
mod = lm(y~x, data)
summary(mod)