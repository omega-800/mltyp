#import "../src/mltyp.typ": standard-scaler, polynomial-features, linear-regression, ridge, lasso, logistic-regression, make-pipeline

// exact linear relationship y = 1 + 3*x1 - 2*x2
#let X = ((1.0, 2.0), (2.0, 3.0), (3.0, 1.0), (4.0, 5.0), (5.0, 4.0), (6.0, 2.0))
#let y = (0.0, 1.0, 8.0, 3.0, 8.0, 15.0)

#let near(a, b, eps: 1e-6) = calc.abs(a - b) < eps

// 1. Linear regression recovers exact coefficients (bias first).
#let lin = (linear-regression().fit)(linear-regression(), X, y)
#assert(near(lin.coef.at(0), 1.0))
#assert(near(lin.coef.at(1), 3.0))
#assert(near(lin.coef.at(2), -2.0))
#assert(near((lin.score)(lin, X, y), 1.0, eps: 1e-9))

// 2. StandardScaler -> zero mean, unit std, inverse round-trip.
#let sc0 = standard-scaler()
#let sc = (sc0.fit)(sc0, X)
#let sX = (sc.transform)(sc, X)
#let means = range(sX.at(0).len()).map(j => sX.map(r => r.at(j)).fold(0.0, (a, b) => a + b) / sX.len())
#for j in range(means.len()) {
  assert(near(means.at(j), 0.0, eps: 1e-9))
}
#let back = (sc.inverse-transform)(sc, sX)
#for i in range(X.len()) {
  for j in range(X.at(0).len()) {
    assert(near(back.at(i).at(j), X.at(i).at(j), eps: 1e-9))
  }
}

// 3. PolynomialFeatures degree 2, no bias on row (1,2) -> [x2, x2^2, x1, x1*x2, x1^2]
#let poly = polynomial-features(degree: 2, include-bias: false)
#let px = (poly.fit-transform)(poly, ((1.0, 2.0),))
#assert(px.len() == 1)
#let expected = (2.0, 4.0, 1.0, 2.0, 1.0)
#for j in range(expected.len()) {
  assert(near(px.at(0).at(j), expected.at(j)))
}

// 4. Ridge with alpha -> roughly recovers plane.
#let rd = (ridge(alpha: 1e-6, solver: "closed-form").fit)(ridge(alpha: 1e-6, solver: "closed-form"), X, y)
#assert(near(rd.coef.at(0), 1.0, eps: 1e-4))
#assert(near(rd.coef.at(1), 3.0, eps: 1e-3))

// 5. Lasso (gradient) roughly recovers plane.
#let ls = (lasso(alpha: 1e-6, lr: 0.02, iters: 4000).fit)(lasso(alpha: 1e-6, lr: 0.02, iters: 4000), X, y)
#assert(near((ls.score)(ls, X, y), 1.0, eps: 1e-6))

// 6. Logistic regression: perfectly separable -> accuracy 1 on training.
#let Xc = ((1.0, 2.0), (2.0, 1.0), (2.0, 3.0), (4.0, 2.0), (5.0, 4.0), (3.0, 5.0))
#let yc = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
#let log = (logistic-regression(lr: 0.3, iters: 2000).fit)(logistic-regression(lr: 0.3, iters: 2000), Xc, yc)
#assert((log.score)(log, Xc, yc) == 1.0)
#let proba = (log.predict-proba)(log, ((3.0, 3.0), (5.0, 4.0)))
#for p in proba {
  assert(p > 0.0 and p < 1.0)
}

// 7. Pipeline: scaler + poly (no bias; ridge adds its own) + ridge
#let pipe = (make-pipeline((standard-scaler(), polynomial-features(degree: 2, include-bias: false), ridge(alpha: 1e-6))).fit)(make-pipeline((standard-scaler(), polynomial-features(degree: 2, include-bias: false), ridge(alpha: 1e-6))), X, y)
#assert(near((pipe.score)(pipe, X, y), 1.0, eps: 1e-6))

= All tests passed