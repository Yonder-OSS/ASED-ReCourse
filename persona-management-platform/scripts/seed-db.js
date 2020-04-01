const { connectDatabase } = require('../controllers/connect-database');
const { seedDatabase } = require('../utils/seed');
const retryFunction = require('../utils/retry-function');

const seedNumPerType = (process.env.SEED_NUM) ? parseInt(process.env.SEED_NUM) : 5;

console.log(`Seeding ${seedNumPerType} personas`);
retryFunction(connectDatabase)
  .then(() => seedDatabase())
  .then(() => {
    console.log('successfully seeded database');
    process.exit(0);
  })
  .catch((err) => {
    console.log(`There was a problem seeding the database`);
    console.log(err);
    process.exit(1);
  });

module.exports = app;