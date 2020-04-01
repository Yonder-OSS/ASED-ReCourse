const mongoose = require('mongoose');
const { logger } = require('@new-knowledge/logger');
const { retryFunction } = require('@new-knowledge/http-utils');
mongoose.Promise = global.Promise;

const {
  personaSchema,
  connectionSchema,
} = require('../models/entity-models');

function createConnectionString() {
  const user = process.env.MONGO_USER || 'mongo';
  const pass = process.env.MONGO_PASS || '1234';
  const host = process.env.MONGO_HOST || 'localhost';
  const port = process.env.MONGO_PORT || 27017;
  const database = process.env.MONGO_DATABASE || 'emu'; // register schemas with mongoose
  return `mongodb://${encodeURIComponent(user)}:${encodeURIComponent(pass)}@${host}:${port}/${database}?authSource=admin`;
}

async function connectDatabase() {
  const connectionString = process.env.MONGO_CONN_STRING || createConnectionString();

  await retryFunction(resetModel);

  logger.debug(`connecting to ${connectionString}`);
  await mongoose.connect(connectionString, { useMongoClient: true });

  mongoose.connection.on('error', (err) => {
    logger.error(`🙅 🚫 🙅 🚫 🙅 🚫 🙅 🚫 → ${err.message}`);
    process.exit(42);
  });
};

async function resetModel() {
  // check if old models are already registered, if so delete, then register
  // once registered models can be accessed w/ Entity = mongoose.model('Entity'); ent = new Entity()
    const models = {
    persona: personaSchema,
    connection: connectionSchema
  };
  for (const type in models) {
    if (mongoose.modelNames().includes(type)) {
      if (process.env.DROP_OLD_DATA && !(process.env.NODE_ENV == 'production')) {
        await mongoose.model(type).collection.drop();
        delete mongoose.connection.models[type];
      }
    }
    await mongoose.model(type, models[type]);
  }
};

module.exports = {
  connectDatabase,
  createConnectionString
};
