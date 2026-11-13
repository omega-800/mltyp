#import "mltyp.typ": standard-scaler, min-max-scaler, robust-scaler, polynomial-features, linear-regression, ridge, lasso, logistic-regression, make-pipeline, train-test-split, k-fold, r_squared, accuracy-score, mean-squared-error
#import "deps.typ": *

= mltyp — demo

#let show-head(label, v) = {
  text(weight: "bold", label) + ": " + repr(v)
}

== Linear regression via normal equations
#let X = ((1.0, 2.0), (2.0, 4.0), (3.0, 5.0), (4.0, 4.0), (5.0, 5.0))
#let y = (7.0, 9.0, 11.0, 13.0, 15.0)
#let lin = (linear-regression().fit)(X, y)
#show-head("lin coef", lin.coef)
#show-head("lin pred", (lin.predict)(((6.0, 5.0), (7.0, 8.0))))
#show-head("lin R2", (lin.score)(X, y))

== Ridge (closed form)
#let rd = (ridge(alpha: 0.5).fit)(X, y)
#show-head("ridge coef", rd.coef)
#show-head("ridge R2", (rd.score)(X, y))

== Lasso (proximal gradient)
#let ls = (lasso(alpha: 0.1, iters: 3000, lr: 0.01).fit)(X, y)
#show-head("lasso coef", ls.coef)
#show-head("lasso R2", (ls.score)(X, y))

== Polynomial features + pipeline
#let poly = polynomial-features(degree: 2, include-bias: false)
#let (polyf, _) = (poly.fit-transform)(((1.0, 2.0), (3.0, 4.0)))
#show-head("poly exponents", polyf.exponents)
#show-head("poly X", (polyf.transform)(((1.0, 2.0), (3.0, 4.0))))

#let pipe = (make-pipeline((standard-scaler(), polynomial-features(degree: 2), ridge(alpha: 0.1))).fit)(X, y)
#show-head("pipe pred", (pipe.predict)(((6.0, 5.0), (7.0, 8.0))))
#show-head("pipe R2", (pipe.score)(X, y))

== Logistic regression
#let Xc = ((1.0, 2.0), (2.0, 1.0), (2.0, 3.0), (4.0, 2.0), (5.0, 4.0), (3.0, 5.0))
#let yc = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
#let log = (logistic-regression(lr: 0.3, iters: 2000).fit)(Xc, yc)
#show-head("log coef", log.coef)
#show-head("log proba", (log.predict-proba)(((3.0, 3.0), (5.0, 4.0))))
#show-head("log pred", (log.predict)(Xc))
#show-head("log acc", (log.score)(Xc, yc))

== Scaler round-trip
#let sc = standard-scaler()
#let (scf, sX) = (sc.fit-transform)(X)
#show-head("scaled", sX)
#show-head("inverse", (scf.inverse-transform)(sX))

== MinMax scaler
#let mm = min-max-scaler()
#let (mmf, mmX) = (mm.fit-transform)(X)
#show-head("mm scaled", mmX)
#show-head("mm inverse", (mmf.inverse-transform)(mmX))

== Robust scaler
#let rb = robust-scaler()
#let (rbf, rbX) = (rb.fit-transform)(X)
#show-head("rb scaled", rbX)
#show-head("rb inverse", (rbf.inverse-transform)(rbX))

== Train / test split
#let (_, tt) = train-test-split(X, y, test-size: 0.4, rng: suiji.gen-rng-f(1))
#show-head("train-x", tt.train-x)
#show-head("train-y", tt.train-y)
#show-head("test-x", tt.test-x)
#show-head("test-y", tt.test-y)

== KFold (3 folds on 6 samples)
#let (_, folds) = k-fold(6, splits: 3, shuffle: true, rng: suiji.gen-rng-f(7))
#show-head("kfold", folds)