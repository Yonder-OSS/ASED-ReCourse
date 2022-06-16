const assert = require('assert');
const _ = require('lodash');
const moment = require('moment');

const chai = require('chai');
chai.should();
// const chaiHttp = require('chai-http');
// chai.use(chaiHttp);

const { logger } = require('@new-knowledge/logger');
const {
  createPerson,
  createPlace,
  createEvent,
  createName,
} = require('../utils/');

// const app = require('../app');

async function sleep(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
};

describe('Testing route-controller functions the publishing of entity update audits to kafka \n\n', function () {

  const user = { username: 'craig corcoran' };
  let app;
  before(async function () {
    app = require('../app');
    while (!app.ready) {
      console.log('waiting for app to be ready');
      await sleep(500);
    }

  });

  it("adds entities", async function () {
    const { addEntity, getEntitiesOfType } = require('../controllers/route-controller');

    const personsBefore = (await getEntitiesOfType('person'));
    const idsBefore = personsBefore.map(p => p._id);

    const newPerson = createPerson();
    const entity = await addEntity('person', newPerson, user);

    const personsAfter = (await getEntitiesOfType('person'));
    const idsAfter = personsAfter.map(p => p._id);

    const newId = personsAfter[0]._id;
    idsBefore.includes(newId).should.be.false;
    idsAfter.includes(newId).should.be.true;

    personsAfter[0].names[0].firstName.should.equal(newPerson.names[0].firstName);
    personsAfter.length.should.equal(personsBefore.length + 1);
    entity.names[0]._id.should.not.be.undefined; // check that subdocs of entity returned have generated ids
  });

  it("adds entities", async function () {
    const { addEntity, getEntitiesOfType } = require('../controllers/route-controller');

    const personsBefore = (await getEntitiesOfType('person'));
    const idsBefore = personsBefore.map(p => p._id);

    const newPerson = createPerson();
    const entity = await addEntity('person', newPerson, user);

    const personsAfter = (await getEntitiesOfType('person'));
    const idsAfter = personsAfter.map(p => p._id);

    const newId = personsAfter[0]._id;
    idsBefore.includes(newId).should.be.false;
    idsAfter.includes(newId).should.be.true;

    personsAfter[0].names[0].firstName.should.equal(newPerson.names[0].firstName);
    personsAfter.length.should.equal(personsBefore.length + 1);
    entity.names[0]._id.should.not.be.undefined; // check that subdocs of entity returned have generated ids
  });

});
