const assert = require('assert');
const chai = require('chai');
const chaiHttp = require('chai-http');
//const { seedDatabase, initServer } = require('../utils/seed');
const { getAllPersonas } = require('../controllers');
const _ = require('lodash');
const mongoose = require('mongoose');
const {
  createName,
  createSocialProfile,
  createPersona, 
  typeToCreateFunc,
  comparePersonas,
  sleep,
} = require('./tools');

chai.use(chaiHttp);
chai.should();

const type = 'persona'

describe('Routing \n\n', function () {

  this.timeout(5000);

  let app;
  const seedNumPerType = parseInt(process.env.SEED_NUM);
  before(async function () {
    console.log('init app');
    app = require('../app');
    while (!app.ready) {
      console.log('waiting for app');
      await sleep(1000);
    }

  });

  after(async function () {
    app.server.close();
  });

  // router.get('/', function(req, res)
  it("should be able to make get req to '/' ", async function () {
    console.log('running basic get test');
    const res = await chai.request(app).get('/');
    res.should.have.status(200);
    res.text.should.equal('Persona Management Platform');
  });

  // router.get(`/personas`, catchReqError(getAllPersonas));  // GET all personas of all type
  it("should be able to make get req to '/personas' and get a list of all personas, sorted", async function () {
    console.log('running get all personas test')
    const res = await chai.request(app).get('/personas?sort=createdAt');
    res.should.have.status(200);

    const personas = JSON.parse(res.text);
    personas.should.have.length(seedNumPerType);

    (new Set(personas.map(e => e.type))).size.should.equal(1); // should include all persona type

    const sorted = _.sortBy(personas, e => e.createdAt);
    _.isEqual(personas, sorted).should.be.true;
  });

  // router.get(`/:id`, catchReqError(getPersonaById)); // GET a single persona by mongo _id
  it("should be able to make get req to '/:id' and get the persona identified", async function () {
    async function testGetPersonaById(type) {
      console.log('running get individual persona by id test')
      const personas = JSON.parse((await chai.request(app).get(`/${type + 's'}`)).text);
      const selected = _.sample(personas);
      const res = await chai.request(app).get(`/${selected._id}`);
      res.should.have.status(200);
      const returned = JSON.parse(res.text);
      _.isEqual(returned, selected).should.be.true;
    }
    await testGetPersonaById(type);
  });

  // router.get(`/:id/:element_id/weights`, catchReqError(getPersonaWeights)); // GET a single persona's weights by mongo _id and element_id
  it("should be able to make get req to '/:id/:element_id/weights' and get the weights of the persona's socialProfile identified", async function () {
    async function testGetWeightsById(type) {
      console.log('running get weights by mongo_id and element_id test')
      const personas = JSON.parse((await chai.request(app).get(`/${type + 's'}`)).text);
      const selected = _.sample(personas);
      const socialProfile = _.sample(selected.socialProfiles)
      const res = await chai.request(app).get(`/${selected._id}/${socialProfile._id}/weights`);
      res.should.have.status(200);
      const returned = JSON.parse(res.text);
      _.isEqual(returned, socialProfile).should.be.true;
    }
    await testGetWeightsById(type);
  });

  // router.post(``, catchReqError(addPersona)); // POST a new persona 
  it("should be able to make post req to '' and add a persona", async function () {
    async function testAddPersona(type) {
      console.log('running post new persona test')
      const newPersona = mongoose.model(type)();
      const res = await chai.request(app).post(``).send({ json: JSON.stringify(newPersona) });
      res.should.have.status(200);
      const returned = JSON.parse(res.text);
      (newPersona._id == returned._id).should.be.true;
      for (var i = 0; i < newPersona.connections.length; i++) {
        (newPersona.connections[i] == returned.connections[i]).should.be.true;
      }
      for (var i = 0; i < newPersona.socialProfiles.length; i++) {
        (newPersona.socialProfiles[i] == returned.socialProfiles[i]).should.be.true;
      }
      for (var i = 0; i < newPersona.names.length; i++) {
        (newPersona.names[i] == returned.names[i]).should.be.true;
      }
    };
    await testAddPersona(type);
  });

  async function addPersona(persona) {
    const response = await chai.request(app).post(``).send({ json: JSON.stringify(persona) });
    return JSON.parse(response.text);
  }

  async function getPersona(id) {
    const response = await chai.request(app).get(`/${id}`);
    return JSON.parse(response.text);
  }

  // POST (`/:fromEntityId/addConnection` (addPersonaField). add a new connection to the fromPersona
  it("should be able to make post req to '/:fromEntityId/addConnection' to add a connection between two personas", async function () {

    async function testAddConnection(fromPersona, type) {
      console.log('test adding connection between personas')
      const toPersona = await addPersona(mongoose.model(type)());
      const body = {
        json: JSON.stringify({toEntityId: toPersona._id }),
      };

      const fromPersonaAfter = (await chai.request(app).post(`/${fromPersona._id}/addConnection`).send(body)).body;
      // const fromPersonaAfter = await getEntity(fromPersona._id);

      const connections = fromPersonaAfter.connections;
      const connIndex = connections.findIndex(c => (c.toEntityId == toPersona._id));
      connIndex.should.not.equal(-1);
      const connection = connections[connIndex];
      connection.should.include({ toEntityId: toPersona._id });
    };

    for (const fromType of type) {
      const fromPersona = await addPersona(mongoose.model(type)());
      await testAddConnection(fromPersona, type);
    }
  });

  ///// NOTE: tests below are not up to date

  // router.post(`/:id/:field`, catchReqError(addPersonaField)); // POST a new element to top-level field or add element to and array-field of persona
  it("should be able to make post req to '/:id/:field' where field is an arrayField and add an element to the array", async function () {
    const testAddArrayField = async (type, primary) => {
      console.log('running test to add element to an array-field')
      const persona = mongoose.model(type)();
      const resBefore = await chai.request(app).post(``).send({
        persona,
      });
      const personaBefore = JSON.parse(resBefore.text);

      const newName = createName(primary);
      const resAfter = await chai.request(app).post(`/${personaBefore._id}/names`).send(newName);
      const personaAfter = JSON.parse(resAfter.text);

      personaAfter._id.should.equal(personaBefore._id); // should be the same persona
      personaBefore.names.length.should.equal(personaAfter.names.length - 1);
      const arrayName = (primary) ? personaAfter.names[0] : _.last(personaAfter.names);
      comparePersonas(newName, arrayName).should.be.true;
    };
    for (const primary of [true, false]) {
      testAddArrayField(type, primary);
    }
  });

  // router.post(`/:id/:field`, catchReqError(addPersonaField)); // POST a new element to top-level field or add element to and array-field of persona
  it("should be able to make post req to '/:id/:field' where field is top-level and set that field", async function () {
    const testAddField = async (type, preExists) => {
      console.log('running test to add an array-field')
      let persona = (preExists) ? { rfi: 'not cool :(' } : {};
      const resBefore = await chai.request(app).post(``).send(persona);
      const personaBefore = JSON.parse(resBefore.text);
      const newRfi = 'super cool!';
      const resAfter = await chai.request(app).post(`/${personaBefore._id}/rfi`).send({ rfi: newRfi });
      const personaAfter = JSON.parse(resAfter.text);

      personaAfter._id.should.equal(personaBefore._id); // should be the same persona
      if (preExists) personaBefore.rfi.should.equal('not cool :(');
      else Object.keys(personaBefore).includes('rfi').should.be.false;
      personaAfter.rfi.should.equal(newRfi);
    };
    for (const preExists of [true, false]) {
      testAddField(type, preExists);
    }
  });

  // router.put(`/:id/:field`, catchReqError(updatePersonaField)); 
  it("should be able to make a put request to '/:id/:field' and update a top-level field", async function () {
    const testUpdatePersonaField = async (type) => {
      console.log('running test to update field of persona')
      const pers = mongoose.model(type)();
      const res = await chai.request(app).post(``).send(pers);
      const persona = JSON.parse(res.text);
      (pers._id == persona._id).should.be.true;
      for (var i = 0; i < pers.connections.length; i++) {
        (pers.connections[i] == persona.connections[i]).should.be.true;
      }
      for (var i = 0; i < pers.socialProfiles.length; i++) {
        (pers.socialProfiles[i] == persona.socialProfiles[i]).should.be.true;
      }
      for (var i = 0; i < pers.names.length; i++) {
        (pers.names[i] == persona.names[i]).should.be.true;
      }

      const update = { names: createName() };
      const updateRes = await chai.request(app).put(`/${persona._id}/names`).send(update);
      const updatePersona = JSON.parse(updateRes.text);
      update.names.firstName.should.equal(updatePersona.names[0].firstName);
      update.names.middleName.should.equal(updatePersona.names[0].middleName);
      update.names.lastName.should.equal(updatePersona.names[0].lastName);
      update.names.should.not.equal(persona.names);
    };
    testUpdatePersonaField(type);
  });

  // /:id/:field - delete persona field
  it("should be able to make a delete request to '/:id/:field' and delete that field", async function () {
    const testDeleteField = async (type) => {
      console.log('running test to delete persona field')
      const pers = mongoose.model(type)();
      const res = await chai.request(app).post(``).send(pers);
      const persona = JSON.parse(res.text);

      const response = await chai.request(app).delete(`/${persona._id}/weights`);
      const retPersona = JSON.parse(response.text);
      assert(retPersona.rfi == undefined);
      comparePersonas(retPersona, persona).should.be.true;
    };
    testDeleteField(type);
  });

  // /:id/:field/:elementId - delete persona array field member
  it("should be able to make a delete request to '/:id/:field/:elementId' and remove that element from an array field", async function () {
    const testDeleteElement = async (type) => {
      console.log('running test to delete persona array field')
      var pers = mongoose.model(type)();
      const res = await chai.request(app).post(``).send(pers);
      const persona = JSON.parse(res.text);
      const update = { names: createName() };
      const updateRes = await chai.request(app).put(`/${persona._id}/names`).send(update);
      const updatePersona = JSON.parse(updateRes.text);
      const removeId = updatePersona.names[0]._id;
      const response = await chai.request(app).delete(`/${updatePersona._id}/names/${removeId}`);
      const retPersona = JSON.parse(response.text);
      const ids = retPersona.names.map(n => n._id);
      ids.includes(removeId).should.be.false;
    };
    testDeleteElement(type);
  });

});
