#import "mltyp.typ": standard-scaler, min-max-scaler, robust-scaler, polynomial-features, linear-regression, ridge, lasso, logistic-regression, make-pipeline, train-test-split, k-fold, r_squared, accuracy-score, mean-squared-error
#import "deps.typ": *

= mltyp — demo

#let show-head(label, v) = {
  text(weight: "bold", label) + ": " + repr(v)
}

== Linear regression via normal equations
#let X = ((1.0, 2.0), (2.0, 4.0), (3.0, 5.0), (4.0, 4.0), (5.0, 5.0))
#let y = (7.0, 9.0, 11.0, 13.0, 15.0)
#let lin = (linear-regression().fit)(linear-regression(), X, y)
#show-head("lin coef", lin.coef)
#show-head("lin pred", (lin.predict)(lin, ((6.0, 5.0), (7.0, 8.0))))
#show-head("lin R2", (lin.score)(lin, X, y))

== Ridge (closed form)
#let rd = (ridge(alpha: 0.5).fit)(ridge(alpha: 0.5), X, y)
#show-head("ridge coef", rd.coef)
#show-head("ridge R2", (rd.score)(rd, X, y))

== Lasso (proximal gradient)
#let ls = (lasso(alpha: 0.1, iters: 3000, lr: 0.01).fit)(lasso(alpha: 0.1, iters: 3000, lr: 0.01), X, y)
#show-head("lasso coef", ls.coef)
#show-head("lasso R2", (ls.score)(ls, X, y))

== Polynomial features + pipeline
#let poly = polynomial-features(degree: 2, include-bias: false)
#show-head("poly exponents", (poly.fit)(poly, X).exponents)
#show-head("poly X", (poly.fit-transform)(poly, ((1.0, 2.0), (3.0, 4.0))))

#let pipe = (make-pipeline((standard-scaler(), polynomial-features(degree: 2), ridge(alpha: 0.1))).fit)(make-pipeline((standard-scaler(), polynomial-features(degree: 2), ridge(alpha: 0.1))), X, y)
#show-head("pipe pred", (pipe.predict)(pipe, ((6.0, 5.0), (7.0, 8.0))))
#show-head("pipe R2", (pipe.score)(pipe, X, y))

== Logistic regression
#let Xc = ((1.0, 2.0), (2.0, 1.0), (2.0, 3.0), (4.0, 2.0), (5.0, 4.0), (3.0, 5.0))
#let yc = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
#let log = (logistic-regression(lr: 0.3, iters: 2000).fit)(logistic-regression(lr: 0.3, iters: 2000), Xc, yc)
#show-head("log coef", log.coef)
#show-head("log proba", (log.predict-proba)(log, ((3.0, 3.0), (5.0, 4.0))))
#show-head("log pred", (log.predict)(log, Xc))
#show-head("log acc", (log.score)(log, Xc, yc))

== Scaler round-trip
#let sc0 = standard-scaler()
#let sc = (sc0.fit)(sc0, X)
#let sX = (sc.transform)(sc, X)
#show-head("scaled", sX)
#show-head("inverse", (sc.inverse-transform)(sc, sX))

== MinMax scaler
#let mm0 = min-max-scaler()
#let mm = (mm0.fit)(mm0, X)
#show-head("mm scaled", (mm.transform)(mm, X))
#show-head("mm inverse", (mm.inverse-transform)(mm, (mm.transform)(mm, X)))

== Robust scaler
#let rb0 = robust-scaler()
#let rb = (rb0.fit)(rb0, X)
#show-head("rb scaled", (rb.transform)(rb, X))
#show-head("rb inverse", (rb.inverse-transform)(rb, (rb.transform)(rb, X)))

== Train / test split
#let (_, tt) = train-test-split(X, y, test-size: 0.4, rng: suiji.gen-rng-f(1))
#show-head("train-x", tt.train-x)
#show-head("train-y", tt.train-y)
#show-head("test-x", tt.test-x)
#show-head("test-y", tt.test-y)

== KFold (3 folds on 6 samples)
#let (_, folds) = k-fold(6, splits: 3, shuffle: true, rng: suiji.gen-rng-f(7))
#show-head("kfold", folds)