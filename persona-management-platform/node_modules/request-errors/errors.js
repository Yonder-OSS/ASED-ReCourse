function catchReqErrorSync(fn) {
  return function (req, res, next) {
    try {
      return fn(req, res, next);
    } catch (err) {
      return reqErrorHandler(err, req, res);
    }
  };
}


function catchReqError(fn) {
  return function (req, res, next) {
    fn(req, res, next).catch(err => reqErrorHandler(err, req, res));
  };
}


function reqErrorHandler(err, req, res) {
  const error = flattenProps(err);
  if (Object.keys(knownErrors).includes(error.name)) {
    console.log('known error: ', error);
    return res.status(error.status || 400).json(error);
  } else {
    console.log('unknown error: ', error);
    return res.status(error.status || 500).json(error);
  }
}


function zipObject(props, vals) {
  const obj = {};
  for (let i = 0; i < props.length; i++) {
    obj[props[i]] = vals[i];
  }
  return obj;
}


function flattenProps(obj) {
  /* returns a cloned object with all properties at top level,
  allowing them to be preserved by JSON.stringify */
  const props = Object.getOwnPropertyNames(obj);
  const vals = props.map(p => obj[p]);
  return zipObject(props, vals);
};


function notFound(req, res, next) {
  next(new PathNotFoundError(`Path ${req.url} Not Found`));
};


class RequestError extends Error {
  constructor(message) {
    super(message);
    this.status = super.status || 500;
    this.name = this.constructor.name;
    if (typeof Error.captureStackTrace === 'function') {
      Error.captureStackTrace(this, this.constructor);
    } else {
      this.stack = (new Error(message)).stack;
    }
  }
}


class InvalidRequestError extends RequestError {
  constructor(message) {
    super(message);
    this.status = 400;
  }
}


class PathNotFoundError extends RequestError {
  constructor(message) {
    super(message);
    this.status = 404;
  }
}


const knownErrors = {
  InvalidRequestError,
  PathNotFoundError,
};


module.exports = {
  ...knownErrors,
  knownErrors,
  reqErrorHandler,
  catchReqErrorSync,
  catchReqError,
  notFound,
};
