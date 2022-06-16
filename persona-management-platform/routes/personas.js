const express = require('express');
const {
  catchReqError,
  InvalidRequestError
} = require('request-errors');
const pluralize = require('pluralize');
pluralize.addIrregularRule('persona', 'personas')

const {
  addFieldOrElement,
  addPersona,
  addPersonaConnection,
  getBatchInfoById,
  deleteArrayElement,
  deletePersona,
  deleteField,
  getAllPersonas,
  getPersona,
  getPersonaWeights,
  editField,
} = require('../controllers/route-controller');

const router = express.Router();

router.get(`/personas`, catchReqError(reqAllPersonas)); // GET all personas of all types
router.get(`/:personaId`, catchReqError(reqPersona)); // GET a single persona by mongo _id
router.get(`/:personaId/:elementId/weights`, catchReqError(reqPersonaWeights));// GET a single persona's socialProfiles weights by mongo _id and element _id

router.post('/getBatchInfo', catchReqError(reqBatchInfo)); // POST a list of batch requests for persona info

router.post(``, catchReqError(reqAddPersona)); // POST a new persona
router.post(`/:fromPersonaId/addConnection`, catchReqError(reqAddPersonaConnection));
router.post(`/:personaId/:field`, catchReqError(reqAddFieldOrElement)); // POST a new element to top-level field or add element to an array-field of persona

router.put(`/:personaId/:field`, catchReqError(reqEditField));

router.delete(`/:personaId`, catchReqError(reqDeletePersona));
router.delete(`/:personaId/:field`, catchReqError(reqDeleteField));
router.delete(`/:personaId/:field/:elementId`, catchReqError(reqDeleteArrayElement));

function unpackBody(req) {
  if (req.body.entity) return req.body.entity;
  if (req.body.payload) return req.body.payload;
  if (req.body.json) return JSON.parse(req.body.json);
  else return req.body;
}

// GET /personas
async function reqAllPersonas(req, res) {
  /* returns all entities sorted by optional query param 'sort'*/
  const allPersonas = await getAllPersonas(req.query.sort);
  return res.json(allPersonas);
};

// GET /:id
async function reqPersona(req, res) {
  /* return persona of given type with given mongo id */
  const persona = await getPersona(
    req.params.personaId);
  return res.json(persona);
};

// GET /:id/weights
async function reqPersonaWeights(req, res) {
  /* return persona of given type with given mongo id */
  const persona = await getPersonaWeights(
    req.params.personaId,
    req.params.elementId);
  return res.json(persona);
};

// POST /
async function reqAddPersona(req, res) {
  /* Adds persona specified in req, returns object as serialized json */
  checkBody(req).catch(err => Promise.reject(err));
  const persona = await addPersona(
    unpackBody(req));
  return res.json(persona);
};

// POST /:fromPersonaId/addConnection w/ body = {payload:{toPersonaId}}
async function reqAddPersonaConnection(req, res) {
  checkBody(req).catch(err => Promise.reject(err));
  const fromPersona = await addPersonaConnection(
    req.params.fromPersonaId,
    unpackBody(req),
  );
  return res.json(fromPersona);
};

// POST /:id/:field
async function reqAddFieldOrElement(req, res) {
  checkBody(req).catch(err => Promise.reject(err));
  const persona = await addFieldOrElement(
    req.params.personaId,
    req.params.field,
    unpackBody(req),
  );
  res.json(persona);
}

// PUT /:id/:field
async function reqEditField(req, res) {
  /* update a top-level field w/ request given mongo id */
  await checkBody(req).catch(err => Promise.reject(err));
  const persona = await editField(
    req.params.personaId,
    req.params.field,
    unpackBody(req),
  );
  return res.json(persona);
};

// DELETE /:personaId  - delete persona by id
async function reqDeletePersona(req, res) {
  const persona = await deletePersona(
    req.params.personaId,
  );
  return res.json(persona);
}

// DELETE /:entityId/:field - delete entity field
async function reqDeleteField(req, res) {
  const persona = await deleteField(
    req.params.personaId,
    req.params.field,
  );
  return res.json(persona);
}

// DELETE /:entityId/:field/:elementId - delete entity array field element
async function reqDeleteArrayElement(req, res) {
  const persona = await deleteArrayElement(
    req.params.personaId,
    req.params.field,
    req.params.elementId,
  );
  return res.json(persona);
}

// TODO assert request is of the correct form
// TODO support subprops (e.g. primaryOfNames.fullName)
async function reqBatchInfo(req, res) {
  const response = {};
  const requestsByType = unpackBody(req); // expects body of form { ids: props }
  for (const id of Object.keys(requestsByType)) {
    const props = requestsByType[id];
    response[id] = await getBatchInfoById(id, props[id] || props);
    // returns object of the form {id : {prop0: value0, prop1: value1}}
  }
  res.json(response);
}

function checkBody(req) {
  // if (!req.body.json && !req.body.entity) return Promise.reject(new InvalidRequestError('Missing payload in "json" or "entity" property of request body'));
  // if (!req.body.user) return Promise.reject(new InvalidRequestError('Missing user property in request body'));
  return Promise.resolve();
}

module.exports = router;
