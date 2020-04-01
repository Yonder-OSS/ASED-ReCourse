const gen = require('./create-components');
const _ = require('lodash');

function createPersona() {
  return {
    names: gen.createName(),
    socialProfiles: gen.createSocialProfile(),
  };
};

const typeToCreateFunc = {
  persona: createPersona,
};

module.exports = {
  createPersona,
  typeToCreateFunc,
};