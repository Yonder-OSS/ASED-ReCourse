const _ = require('lodash');
const moment = require('moment');
const {
  createName,
  typeToCreateFunc
} = require('../utils/create-components');
const {
  createPersona,
} = require('../utils/create-entities');

const compareTimes = (string1, string2) => {
  if (isNaN(Date.parse(string1)) || isNaN(Date.parse(string2))) return false;
  else return moment(string1).isSame(string2, 'minute');
};

const compareStrings = (string1, string2) => {
  if (string1 == string2) return true;
  else {
    return compareTimes(string1, string2);
  }
};

const compareElements = (el1, el2) => {
  if (el1 == el2) return true;
  if (_.isEqual(el1, el2)) return true;
  if ((typeof el1 == 'string') && (typeof el2 == 'string')) {
    return compareStrings(el1, el2);
  }
  return false;
};

const comparePersonas = (sent, retr) => {
  if (_.isEqual(sent, retr)) return true;

  const personaKeys = Object.keys(sent);
  const matching = personaKeys.map(k => {
    if (sent[k] instanceof Array) {
      return _.range(sent[k].length).every(i => {
        const sentEl = sent[k][i];
        const retrEl = retr[k][i];
        const elementKeys = Object.keys(sentEl);
        return elementKeys.every(ek => {
          return compareElements(sentEl[ek], retrEl[ek]);
        });
      });
    } else if ((typeof sent[k] == 'object') && (typeof retr[k] == 'object')) {
      const objKeys = Object.keys(sent[k]);
      return objKeys.every(ok => _.isEqual(sent[k][ok], retr[k][ok]));
    } else {
      return compareElements(sent[k], retr[k]);
    }
  });
  return matching.every(k => k);
};

function sleep(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
};

module.exports = {
  comparePersonas,
  typeToCreateFunc,
  createName,
  sleep,
};