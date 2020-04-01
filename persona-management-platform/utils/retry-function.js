const promiseRetry = require('promise-retry');
const { logger } = require('@new-knowledge/logger');

const numRetries = process.env.NUM_RETRIES || 20;
const retryOptions = { retries: numRetries };

function retryFunction(func, args) {
  logger.debug(`attempting to call function ${func.name}` + ((args) ? `with arguments ${args}` : ''));

  return promiseRetry(retryOptions, function (retry, number) {
    logger.debug(`attempt to call ${func.name} number ${number} of ${numRetries}`);
    return func(args)
      .catch(err => {
        logger.debug(`failed attempt to call ${func.name}, error: ${err}`); //.message}`);
        retry(err);
      });

  })
    .then(() => {
      logger.info(`completed successful call of ${func.name}` + (args ? ` with arguments ${args}` : ''));
    })
    .catch(err => {
      logger.error(`failed all attempts to call ${func.name}` + (args ? ` with arguments ${args}` : '') + `, error: ${err}`);
      throw err;
    });
}

module.exports = retryFunction;