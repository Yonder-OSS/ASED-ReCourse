const _ = require('lodash');
const moment = require('moment');
const faker = require('faker');

let { socialProfileTypes } = require('./metadata');
socialProfileTypes = socialProfileTypes.map(t => t.value);

function randString() {
  return Math.random().toString(36).substring(7);
};


function createName(primary = false) {
  return {
    primary,
    firstName: faker.name.firstName(),
    middleName: faker.name.firstName(),
    lastName: faker.name.lastName(),
  };
}

// TODO use real profiles here to test scraping?
function createSocialProfile() {
  return {
    name: faker.internet.userName(),

    type: _.sample(socialProfileTypes),
    url: faker.internet.url(),
    filename: "ppdb-l2018-11-26_16-41-17_200-0.6008.h5",
    collect: true,
  };
};


// TODO use valid entity Ids for toEntityId
function createConnection() {
  return {
    toEntityId: faker.random.uuid(), // TODO valid mongo ids
  };
}

module.exports = {
  createConnection,
  createName,
  createSocialProfile,
};
