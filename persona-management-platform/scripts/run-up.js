const { up } = require('./type-migration');

/**
 * Migration to latest schema
 */
(async () => {
  await up();
  process.exit(0);
  return;
})();
