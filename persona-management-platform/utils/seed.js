const _ = require('lodash');
const { logger } = require('@new-knowledge/logger');
const request = require('request-promise');
const chai = require('chai');
const chaiHttp = require('chai-http');
const fs = require('fs');

const { createPersona } = require('./create-entities');

chai.use(chaiHttp);

async function seed(seedNumPerType = 5, app) {
  logger.debug('seeding persona database');
  try {
    for (const i of _.range(seedNumPerType)) {

      const entityParams = createPersona();
      const body = {
        json: JSON.stringify(entityParams),
        //user: { username: 'craig' }
      };

      if (app) {
        await chai.request(app).post(``).send(body);
      } else {
        await request.post(`http://emu:3000/`, { body, json: true });
      }
    }
  } catch (err) {
    // logger.error(`seeding error: ${err}`);
    console.log(`seeding error: `, err);
    throw err;
  }
};

// wipe data
async function wipe() {
  // Wipe DB by downing and deleting volumes
};

// TODO support varying number of entities seeded by type
async function seedDatabase(seedNumPerType, app) {

  seedNumPerType = seedNumPerType || (process.env.SEED_NUM) ? parseInt(process.env.SEED_NUM) : 5;
  if (isNaN(seedNumPerType)) {
    logger.error('SEED_NUM environment variable not an int, skipping seed');
  }

  await seed(seedNumPerType, app);
}

module.exports = {
  seedDatabase,
};