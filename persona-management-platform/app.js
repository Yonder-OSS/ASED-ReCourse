const express = require('express');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const fs = require('fs');
mongoose.Promise = global.Promise;

const { seedDatabase } = require('./utils/seed');

const { loggingMiddleware, logger } = require('@new-knowledge/logger');
const { notFound, reqErrorHandler } = require('request-errors');

const index = require('./routes');
const {
  initServer,
  retryFunction,
} = require('./utils');

const {
  connectDatabase,
} = require('./controllers');

process.on('unhandledRejection', (reason) => {
  // logger.error(`promise rejection: ${JSON.stringify(reason)}`);
  console.log('promise rejection:', reason);
  throw reason;
});

const app = express();

app.server = initServer(app);

app.use(loggingMiddleware);

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());

retryFunction(connectDatabase)
  .then(() => {
    app.use('/', index);
    app.use(notFound);
    app.use(reqErrorHandler);
  })
  .then(() => {
    app.ready = true; // currently just for testing
    logger.debug('service ready');
    seedDatabase(1, app);
  })
  .catch(err => {
    logger.debug(`failed to connect to and initialize entity db and associated consumers/producers, error: ${err}`);
  });

module.exports = app;
