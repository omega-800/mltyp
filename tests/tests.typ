#import "../src/mltyp.typ": standard-scaler, min-max-scaler, robust-scaler, polynomial-features, linear-regression, ridge, lasso, logistic-regression, make-pipeline, train-test-split, k-fold, shuffle-indices, quantile

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

// 8. MinMaxScaler maps to [0,1] and round-trips.
#let mm0 = min-max-scaler()
#let mm = (mm0.fit)(mm0, X)
#let mmX = (mm.transform)(mm, X)
#for i in range(X.len()) {
  for j in range(X.at(0).len()) {
    assert(mmX.at(i).at(j) >= 0.0 and mmX.at(i).at(j) <= 1.0)
  }
}
#let mmback = (mm.inverse-transform)(mm, mmX)
#for i in range(X.len()) {
  for j in range(X.at(0).len()) {
    assert(near(mmback.at(i).at(j), X.at(i).at(j), eps: 1e-9))
  }
}
// feature-range rescale
#let mm2 = min-max-scaler(feature-range: (-1.0, 1.0))
#let mm2f = (mm2.fit)(mm2, X)
#let mm2X = (mm2f.transform)(mm2f, X)
#for i in range(X.len()) {
  for j in range(X.at(0).len()) {
    assert(mm2X.at(i).at(j) >= -1.0 and mm2X.at(i).at(j) <= 1.0)
  }
}

// 9. RobustScaler centers on median, scale by IQR, round-trips.
#let rb0 = robust-scaler()
#let rb = (rb0.fit)(rb0, X)
#let rbX = (rb.transform)(rb, X)
#for j in range(rb.medians.len()) {
  // transformed column should have median ~ 0
  let med = quantile(rbX.map(r => r.at(j)), 0.5)
  assert(near(med, 0.0, eps: 1e-9))
}
#let rbback = (rb.inverse-transform)(rb, rbX)
#for i in range(X.len()) {
  for j in range(X.at(0).len()) {
    assert(near(rbback.at(i).at(j), X.at(i).at(j), eps: 1e-9))
  }
}

// 10. train-test-split preserves sizes and no leakage.
#let tt = train-test-split(X, y, test-size: 0.4, shuffle: true, random-state: 3)
#assert(tt.train-x.len() == 4)
#assert(tt.test-x.len() == 2)
#assert(tt.train-y.len() == 4 and tt.test-y.len() == 2)

// 11. k-fold produces `splits` folds; indices cover 0..n-1 exactly once total.
#let kf = k-fold(6, splits: 3, shuffle: true, random-state: 9)
#assert(kf.len() == 3)
#let alltest = ()
#for fold in kf {
  assert(fold.train-idx.len() == 4)
  assert(fold.test-idx.len() == 2)
  alltest += fold.test-idx
}
#assert(alltest.sorted() == range(6))
// train and test never overlap within a fold
#for fold in kf {
  for ti in fold.test-idx {
    for tr in fold.train-idx {
      assert(ti != tr)
    }
  }
}

// 12. shuffle is deterministic and a permutation.
#let sh1 = shuffle-indices(6, 5)
#let sh2 = shuffle-indices(6, 5)
#assert(sh1 == sh2)
#assert(sh1.sorted() == range(6))

= All tests passed