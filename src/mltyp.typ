#import "deps.typ": *
// mltyp — a small scikit-learn style machine-learning library written in
// pure Typst, using only standard library builtins.
//
// Conventions:
//  * A data matrix X is an array of rows, each row an array of numbers.
//  * A label vector y is an array of numbers.
//  * Estimators are dictionaries with methods. Because Typst has no
//    classes, every method takes the estimator itself as its first
//    argument (`self`) and returns a *new* estimator when it mutates
//    state. Objects are immutable; fitting returns an updated copy.
//  * Transformers expose `fit`, `transform`, `fit-transform`.
//  * Estimators expose `fit`, `predict` (and `score`).

// ----------------------------------------------------------------------------
// Low-level vector / matrix helpers
// ----------------------------------------------------------------------------

#let vec-sum(a) = a.fold(0.0, (acc, v) => acc + v)

#let vec-mean(a) = if a.len() == 0 { 0.0 } else { vec-sum(a) / a.len() }

#let xpow(b, e) = if e == 0 { 1 } else { calc.pow(b, e) }

#let arr-set(a, i, v) = a.slice(0, i) + (v,) + a.slice(i + 1, a.len())

#let dict-update(d, ..rest) = d + rest.named()

#let vec-dot(a, b) = vec-sum(a.zip(b).map(p => p.at(0) * p.at(1)))

#let vec-norm2(a) = vec-sum(a.map(v => v * v))

#let vec-sub(a, b) = a.zip(b).map(p => p.at(0) - p.at(1))

#let vec-add(a, b) = a.zip(b).map(p => p.at(0) + p.at(1))

#let vec-scale(a, s) = a.map(v => v * s)

#let vec-op(a, b, f) = a.zip(b).map(p => f(p.at(0), p.at(1)))

#let m-transpose(m) = {
  if m.len() == 0 {
    ()
  } else {
    range(m.at(0).len()).map(j => m.map(row => row.at(j)))
  }
}

// matrix-matrix product: a (n x k) times b (k x m)
#let m-mul(a, b) = {
  let bt = m-transpose(b)
  a.map(row => bt.map(col => vec-dot(row, col)))
}

// matrix-vector product: a (n x k) times v (k)
#let m-vec(a, v) = a.map(row => vec-dot(row, v))

// transpose then multiply: (a^T * v) where v is a vector
#let mt-vec(a, v) = m-transpose(a).map(row => vec-dot(row, v))

#let m-identity(n) = {
  range(n).map(i => range(n).map(j => if i == j { 1 } else { 0 }))
}

#let m-scale(a, s) = a.map(row => vec-scale(row, s))

#let m-add(a, b) = a.enumerate().map(it => vec-add(it.at(1), b.at(it.at(0))))

#let m-sub(a, b) = a.enumerate().map(it => vec-sub(it.at(1), b.at(it.at(0))))

#let as-matrix(x) = {
  if x.len() > 0 and type(x.at(0)) == array { x } else { (x,) }
}

#let col-means(x) = {
  if x.len() == 0 {
    ()
  } else {
    range(x.at(0).len()).map(j => vec-mean(x.map(row => row.at(j))))
  }
}

// ----------------------------------------------------------------------------
// Scoring metrics
// ----------------------------------------------------------------------------

#let mean-squared-error(ytrue, ypred) = {
  let n = ytrue.len()
  vec-sum(ytrue.zip(ypred).map(p => (p.at(0) - p.at(1)) * (p.at(0) - p.at(1)))) / n
}

#let r_squared(ypred, ytrue) = {
  let mu = vec-mean(ytrue)
  let ss-res = vec-sum(ytrue.zip(ypred).map(p => (p.at(0) - p.at(1)) * (p.at(0) - p.at(1))))
  let ss-tot = vec-sum(ytrue.map(v => (v - mu) * (v - mu)))
  if ss-tot == 0 { 0 } else { 1 - ss-res / ss-tot }
}

#let accuracy-score(ypred, ytrue) = {
  let n = ytrue.len()
  ytrue.zip(ypred).filter(p => p.at(0) == p.at(1)).len() / n
}

// Solve A x = b for a square n x n matrix A via Gaussian elimination with
// partial pivoting. Returns the solution vector x (length n).
#let gauss-solve(A, b) = {
  let n = A.len()
  if n == 0 {
    return ()
  }
  // augmented matrix
  let M = range(n).map(i => A.at(i) + (b.at(i),))

  // forward elimination
  for col in range(n) {
    let pivot = col
    for r in range(col + 1, n) {
      if calc.abs(M.at(r).at(col)) > calc.abs(M.at(pivot).at(col)) {
        pivot = r
      }
    }
    M = arr-set(arr-set(M, col, M.at(pivot)), pivot, M.at(col))
    for r in range(col + 1, n) {
      let f = M.at(r).at(col) / M.at(col).at(col)
      M = arr-set(M, r, M.at(r).enumerate().map(it => {
        let v = it.at(1)
        if it.at(0) == col { 0 } else { v - f * M.at(col).at(it.at(0)) }
      }))
    }
  }

  // back substitution
  let x = range(n).map(v => 0.0)
  for i in range(n) {
    let row = n - 1 - i
    let acc = M.at(row).at(n)
    for j in range(row + 1, n) {
      acc -= M.at(row).at(j) * x.at(j)
    }
    x = arr-set(x, row, acc / M.at(row).at(row))
  }
  x
}

// Ridge-regression closed-form solution (with an explicit bias column).
// X is n x k (augmented with a ones column). Returns weight vector.
#let closed-form-ridge(X, y, alpha) = {
  let k = X.at(0).len()
  let A = m-add(m-mul(m-transpose(X), X), m-scale(m-identity(k), alpha))
  let b = mt-vec(X, y)
  gauss-solve(A, b)
}

// ----------------------------------------------------------------------------
// StandardScaler
// ----------------------------------------------------------------------------

#let standard-scaler() = (
  means: (),
  stds: (),
  fit: (self, x) => {
    let n = x.len()
    let means = col-means(x)
    let stds = range(x.at(0).len()).map(j => {
      let mu = means.at(j)
      calc.sqrt(vec-sum(x.map(row => (row.at(j) - mu) * (row.at(j) - mu))) / n)
    })
    dict-update(self, means: means, stds: stds)
  },
  transform: (self, x) =>
    x.map(row => row.zip(self.means, self.stds).map(t =>
      (t.at(0) - t.at(1)) / t.at(2)
    )),
  fit-transform: (self, x) => { let f = (self.fit)(self, x); (f.transform)(f, x) },
  inverse-transform: (self, x) =>
    x.map(row => row.zip(self.means, self.stds).map(t =>
      t.at(0) * t.at(2) + t.at(1)
    )),
)

// ----------------------------------------------------------------------------
// Quantile / min-max helpers used by the scalers
// ----------------------------------------------------------------------------

// Linear-interpolated quantile (q in [0,1]) of an array `a`.
#let quantile(a, q) = {
  let s = a.sorted()
  let n = s.len()
  let p = q * (n - 1)
  let lo = calc.floor(p)
  let hi = calc.ceil(p)
  if lo == hi {
    s.at(lo)
  } else {
    s.at(lo) + (p - lo) * (s.at(hi) - s.at(lo))
  }
}

#let col-extremes(x, pick) = {
  range(x.at(0).len()).map(j =>
    x.map(row => row.at(j)).fold(x.at(0).at(j), pick)
  )
}

// ----------------------------------------------------------------------------
// MinMaxScaler
// ----------------------------------------------------------------------------

#let min-max-scaler(feature-range: (0.0, 1.0)) = (
  feature-range: feature-range,
  mins: (),
  maxs: (),
  fit: (self, x) =>
    dict-update(
      self,
      mins: col-extremes(x, calc.min),
      maxs: col-extremes(x, calc.max),
    ),
  transform: (self, x) => {
    let lo = self.feature-range.at(0)
    let hi = self.feature-range.at(1)
    x.map(row =>
      row.zip(self.mins, self.maxs).map(t =>
        (t.at(0) - t.at(1)) / (t.at(2) - t.at(1)) * (hi - lo) + lo
      )
    )
  },
  fit-transform: (self, x) => { let f = (self.fit)(self, x); (f.transform)(f, x) },
  inverse-transform: (self, x) => {
    let lo = self.feature-range.at(0)
    let hi = self.feature-range.at(1)
    x.map(row =>
      row.zip(self.mins, self.maxs).map(t =>
        (t.at(0) - lo) / (hi - lo) * (t.at(2) - t.at(1)) + t.at(1)
      )
    )
  },
)

// ----------------------------------------------------------------------------
// RobustScaler
// ----------------------------------------------------------------------------

#let robust-scaler(quantile-range: (0.25, 0.75)) = (
  quantile-range: quantile-range,
  medians: (),
  lower: (),
  upper: (),
  fit: (self, x) => {
    let n = x.at(0).len()
    let qlo = self.quantile-range.at(0)
    let qhi = self.quantile-range.at(1)
    dict-update(
      self,
      medians: range(n).map(j => quantile(x.map(row => row.at(j)), 0.5)),
      lower: range(n).map(j => quantile(x.map(row => row.at(j)), qlo)),
      upper: range(n).map(j => quantile(x.map(row => row.at(j)), qhi)),
    )
  },
  transform: (self, x) =>
    x.map(row =>
      row.zip(self.medians, self.lower, self.upper).map(t =>
        (t.at(0) - t.at(1)) / (t.at(3) - t.at(2))
      )
    ),
  fit-transform: (self, x) => { let f = (self.fit)(self, x); (f.transform)(f, x) },
  inverse-transform: (self, x) =>
    x.map(row =>
      row.zip(self.medians, self.lower, self.upper).map(t =>
        t.at(0) * (t.at(3) - t.at(2)) + t.at(1)
      )
    ),
)

// ----------------------------------------------------------------------------
// PolynomialFeatures
// ----------------------------------------------------------------------------

// Enumerate all exponent vectors of length `n` with total degree in the
// allowed range, honoring include-bias and interaction-only.
#let poly-exponents(n, degree, include-bias, interaction-only) = {
  // iterative expansion: states are (vector, sum-of-exponents) pairs
  let states = (( (), 0 ),)
  for i in range(n) {
    let next = ()
    for st in states {
      let vec = st.at(0)
      let sum = st.at(1)
      let hi = if interaction-only { calc.min(1, degree - sum) } else { degree - sum }
      for e in range(hi + 1) {
        next += ((vec + (e,), sum + e),)
      }
    }
    states = next
  }
  states.map(st => st.at(0)).filter(vec => {
    let s = vec-sum(vec)
    let keep-sum = s >= (if interaction-only { 1 } else { 0 }) and s <= degree
    let keep-bias = include-bias or s > 0
    keep-sum and keep-bias
  })
}

#let polynomial-features(degree: 2, include-bias: true, interaction-only: false) = (
  degree: degree,
  include-bias: include-bias,
  interaction-only: interaction-only,
  exponents: (),
  n-features: (),
  fit: (self, x) =>
    dict-update(self, 
      exponents: poly-exponents(
        x.at(0).len(),
        self.degree,
        self.include-bias,
        self.interaction-only,
      ),
      n-features: self.exponents.len(),
    ),
  transform: (self, x) =>
    x.map(row =>
      self.exponents.map(exp =>
        exp.enumerate().map(it => xpow(row.at(it.at(0)), it.at(1))).fold(1, (a, b) => a * b)
      )
    ),
  fit-transform: (self, x) => { let f = (self.fit)(self, x); (f.transform)(f, x) },
)

// ----------------------------------------------------------------------------
// Gradient-descent helpers for linear models
// ----------------------------------------------------------------------------

#let sigmoid(z) = 1.0 / (1.0 + calc.exp(-z))

// Common gradient-descent core for ridge / lasso / logistic. Operates in raw
// feature space (bias column of ones prepended), so the returned `w` is used
// directly by predict-linear. penalty: "ridge" | "lasso" | "logistic".
#let fit-linear(X, y, alpha, lr, iters, penalty) = {
  let aug = X.map(row => (1,) + row)
  let n = aug.len()
  let k = aug.at(0).len()
  let w = range(k).map(v => 0.0)

  for step in range(iters) {
    // pred = X w  (or sigmoid(X w) for logistic)
    let pred = m-vec(aug, w).map(z => if penalty == "logistic" { sigmoid(z) } else { z })
    // residual = pred - y
    let residual = vec-sub(pred, y)
    // gradient of (1/2n)||residual||^2  =>  (1/n) X^T residual
    let grad = vec-scale(mt-vec(aug, residual), 1.0 / n)
    let wnext = vec-sub(w, vec-scale(grad, lr))

    if penalty == "ridge" {
      // shrink all non-bias weights by the L2 penalty (bias at index 0 is free)
      let shrink = 1 - lr * alpha / n
      wnext = wnext.enumerate().map(it => if it.at(0) == 0 { it.at(1) } else { it.at(1) * shrink })
    } else if penalty == "lasso" {
      // proximal (soft-threshold) step on the non-bias weights
      let th = lr * alpha / n
      wnext = wnext.enumerate().map(it => {
        if it.at(0) == 0 {
          it.at(1)
        } else {
          let v = it.at(1)
          if v > th { v - th } else if v < -th { v + th } else { 0 }
        }
      })
    }
    w = wnext
  }
  w
}

#let predict-linear(w, x) = {
  // x is a single row (without bias) or a matrix; w[0] is the bias
  let Xs = as-matrix(x)
  let aug = Xs.map(row => (1,) + row)
  m-vec(aug, w)
}

// ----------------------------------------------------------------------------
// LinearRegression
// ----------------------------------------------------------------------------

#let linear-regression(alpha: 0.0) = (
  alpha: alpha,
  coef: (),
  fit: (self, x, y) => {
    let xa = x.map(row => (1,) + row)
    dict-update(self, coef: closed-form-ridge(xa, y, self.alpha))
  },
  predict: (self, x) => predict-linear(self.coef, x),
  score: (self, x, y) => {
    let pred = predict-linear(self.coef, x)
    r_squared(pred, y)
  },
)

// ----------------------------------------------------------------------------
// Ridge (closed form by default, gradient descent optional)
// ----------------------------------------------------------------------------

#let ridge(alpha: 1.0, solver: "closed-form", lr: 0.1, iters: 200) = (
  alpha: alpha,
  lr: lr,
  iters: iters,
  solver: solver,
  coef: (),
  fit: (self, x, y) => {
    let w = {
      if self.solver == "closed-form" {
        let xa = x.map(row => (1,) + row)
        closed-form-ridge(xa, y, self.alpha)
      } else {
        fit-linear(x, y, self.alpha, self.lr, self.iters, "ridge")
      }
    }
    dict-update(self, coef: w)
  },
  predict: (self, x) => predict-linear(self.coef, x),
  score: (self, x, y) => {
    let pred = predict-linear(self.coef, x)
    r_squared(pred, y)
  },
)

// ----------------------------------------------------------------------------
// Lasso (coordinate descent via proximal gradient / ISTA)
// ----------------------------------------------------------------------------

#let lasso(alpha: 1.0, lr: 0.01, iters: 500) = (
  alpha: alpha,
  lr: lr,
  iters: iters,
  coef: (),
  fit: (self, x, y) =>
    dict-update(self, 
      coef: fit-linear(x, y, self.alpha, self.lr, self.iters, "lasso"),
    ),
  predict: (self, x) => predict-linear(self.coef, x),
  score: (self, x, y) => {
    let pred = predict-linear(self.coef, x)
    r_squared(pred, y)
  },
)

// ----------------------------------------------------------------------------
// LogisticRegression
// ----------------------------------------------------------------------------

#let logistic-regression(alpha: 0.0, lr: 0.1, iters: 300) = (
  alpha: alpha,
  lr: lr,
  iters: iters,
  coef: (),
  fit: (self, x, y) =>
    dict-update(self, 
      coef: fit-linear(x, y, self.alpha, self.lr, self.iters, "logistic"),
    ),
  predict-proba: (self, x) => {
    let Xs = as-matrix(x)
    let aug = Xs.map(row => (1,) + row)
    m-vec(aug, self.coef).map(z => sigmoid(z))
  },
  predict: (self, x) =>
    (self.predict-proba)(self, x).map(p => if p >= 0.5 { 1 } else { 0 }),
  score: (self, x, y) => {
    let pred = (self.predict)(self, x)
    accuracy-score(pred, y)
  },
)

// ----------------------------------------------------------------------------
// Pipelines
// ----------------------------------------------------------------------------

#let make-pipeline(steps) = {
  let transforms = steps.slice(0, steps.len() - 1)
  (
    steps: steps,
    fitted-transforms: (),
    fitted-estimator: none,
    fit: (self, x, y) => {
      let xt = x
      let fitted = ()
      for t in transforms {
        let tf = (t.fit)(t, xt)
        xt = (tf.transform)(tf, xt)
        fitted += (tf,)
      }
      let estimator = steps.at(steps.len() - 1)
      let fe = (estimator.fit)(estimator, xt, y)
      dict-update(self, fitted-transforms: fitted, fitted-estimator: fe)
    },
    _apply: (self, x) => {
      let xt = x
      for t in self.fitted-transforms {
        xt = (t.transform)(t, xt)
      }
      xt
    },
    predict: (self, x) => {
      let xt = (self._apply)(self, x)
      (self.fitted-estimator.predict)(self.fitted-estimator, xt)
    },
    score: (self, x, y) => {
      let xt = (self._apply)(self, x)
      (self.fitted-estimator.score)(self.fitted-estimator, xt, y)
    },
  )
}

// ----------------------------------------------------------------------------
// Model selection: shuffling, train/test split, KFold
//
// Randomness is provided by a suiji rng object.
// Conventions:
//  * Functions that *require* randomness take the rng as their FIRST
//    positional argument: shuffle-indices(rng, n), shuffle(rng, a).
//  * Functions where randomness is *optional* (they have a `shuffle:` flag)
//    take the rng as a named argument `rng:` instead of positionally.
//  * Any function that consumes an rng returns it back as the FIRST element
//    of a tuple: (new-rng, result). Callers thread that rng onward.
// ----------------------------------------------------------------------------

// Shuffle the integers 0..n-1. Returns (new-rng, indices).
#let shuffle-indices(rng, n) = {
  let (rng, idx) = suiji.shuffle-f(rng, range(n))
  (rng, idx)
}

// Shuffle an array. Returns (new-rng, shuffled).
#let shuffle(rng, a) = {
  let (rng, res) = suiji.shuffle-f(rng, a)
  (rng, res)
}

// Split X and y into train and test portions.
// Returns (new-rng, (train-x, train-y, test-x, test-y)).
#let train-test-split(X, y, test-size: 0.25, shuffle: true, rng: none) = {
  let r = rng
  if shuffle and r == none {
    r = suiji.gen-rng-f(0)
  }
  let n = X.len()
  let idx = range(n)
  if shuffle {
    (r, idx) = suiji.shuffle-f(r, idx)
  }
  let ntest = calc.floor(n * test-size)
  let ntrain = n - ntest
  let train-idx = idx.slice(0, ntrain)
  let test-idx = idx.slice(ntrain, n)
  (
    r,
    (
      train-x: train-idx.map(i => X.at(i)),
      train-y: train-idx.map(i => y.at(i)),
      test-x: test-idx.map(i => X.at(i)),
      test-y: test-idx.map(i => y.at(i)),
    ),
  )
}

// Split n samples into `splits` roughly equal folds (round-robin assignment).
// Returns (new-rng, folds), where folds is an array of `splits` dicts, each
// with (train-idx, test-idx).
#let k-fold(n, splits: 5, shuffle: false, rng: none) = {
  let r = rng
  if shuffle and r == none {
    r = suiji.gen-rng-f(0)
  }
  let idx = range(n)
  if shuffle {
    (r, idx) = suiji.shuffle-f(r, idx)
  }
  let folds = range(splits).map(f => {
    let test-idx = ()
    let train-idx = ()
    for i in range(n) {
      if calc.rem(i, splits) == f {
        test-idx += (idx.at(i),)
      } else {
        train-idx += (idx.at(i),)
      }
    }
    (train-idx: train-idx, test-idx: test-idx)
  })
  (r, folds)
}
