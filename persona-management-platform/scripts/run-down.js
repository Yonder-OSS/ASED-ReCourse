const { down } = require('./type-migration');


/**
 * Revert migration to latest schema
 */
(async () => {
  await down();
  process.exit(0);
  return;
})();
