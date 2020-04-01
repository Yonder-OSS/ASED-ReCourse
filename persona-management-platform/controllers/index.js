const eventHandler = require('./event-handler');

const { connectDatabase, createConnectionString } = require('./connect-database');

const {
  addPersona,
  addPersonaConnection,
  addFieldOrElement,
  deleteArrayElement,
  deletePersona,
  deleteField,
  getAllPersonas,
  getBatchInfoById,
  getPersona,
  editField,
  isNewSocialProfile,
  publishNewEntity,
} = require('./route-controller');

module.exports = {
  addPersona,
  addPersonaConnection,
  addFieldOrElement,
  connectDatabase,
  createConnectionString,
  deleteArrayElement,
  deletePersona,
  deleteField,
  editField,
  eventHandler,
  getAllPersonas,
  getBatchInfoById,
  getPersona,
  isNewSocialProfile,
  publishNewEntity,
};
