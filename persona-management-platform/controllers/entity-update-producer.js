const kafka = require('@new-knowledge/nk-kafka-node');
const { logger } = require('@new-knowledge/logger');
const { InvalidRequestError } = require('request-errors');
const _ = require('lodash');

const jsonfile = require('jsonfile');

const viewTopic = process.env.ENTITY_VIEW_TOPIC || 'db.entities.view';
const updateTopic = process.env.ENTITY_UPDATE_TOPIC || 'db.entities.update';

let producer;

async function initUpdateProducer() {
  logger.debug('creating entity update producer');
  producer = await kafka.createProducer();

  producer.on('error', (err) => {
    logger.error(`There was a problem producing a message: ${err.message}. ${err.stack}`);
  });
}

async function publishEntityUpdate(entity, entityType, entityId, updateType, updateEvents, user = {}) {
  entity = removeEmpty(entity._doc || entity);

  if (Array.isArray(updateEvents) && (updateEvents.length > 0)) {
    const doc = {
      entityType,
      entityId,
      updateType,
      username: (typeof user == 'string') ? user : user.username || 'unknown',
    };

    doc.entity = sanitizeDocument(removeEmpty(entity._doc || entity));
    doc.createdAt = new Date().toISOString();

    // make sanitized copies of update event values and
    doc.updateEvents = updateEvents.map(event => {
      const result = { ...event };
      if (event.oldValue && event.oldValue.sanitize) {
        result.oldValue = event.oldValue.toObject(); // need to make a copy here or the mongoose obj will strip non-schema props on serialization (?)
        result.oldValue.sanitized = event.oldValue.sanitize();
      }
      if (event.newValue && event.newValue.sanitize) {
        result.newValue = event.newValue.toObject(); // need to make a copy here or the mongoose obj will strip non-schema props on serialization (?)
        result.newValue.sanitized = event.newValue.sanitize();
      }
      return result;
    });

    // write doc to file if env var set
    if (process.env.LOG_KAFKA_PRODUCER) jsonfile.writeFileSync('test/updates.json', doc, { flag: 'a' });

    return producer.produce(updateTopic, JSON.stringify(doc));

  } else {
    throw new InvalidRequestError('update published with no update events');
  }
}

async function publishEntityView(entityType, entityId, user = {}) {
  const doc = {
    entityType,
    entityId,
    username: (typeof user == 'string') ? user : user.username || 'unknown',
  };

  doc.createdAt = new Date().toISOString();

  // write (append) doc to file if env var set
  if (process.env.LOG_KAFKA_PRODUCER) jsonfile.writeFileSync('test/views.json', doc, { flag: 'a' });

  return producer.produce(viewTopic, JSON.stringify(doc));
}

function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

function sanitizeDocument(doc) {
  const sanitizedDoc = {};
  const exclude = ['__v', '_rev'];
  const docProps = Object.getOwnPropertyNames(doc).filter(prop => !exclude.includes(prop));
  for (const prop of docProps) {
    const propVal = doc[prop];

    if (Array.isArray(propVal)) {
      sanitizedDoc[prop] = propVal.map(el => el.sanitize ? el.sanitize() : el);

      // add primaryOfProps for arrays with primary elements
      if ((propVal.length > 0) && (propVal[0].primary != undefined)) {
        const primaryElement = propVal.find(el => el.primary); // assumes unique primary
        if (!primaryElement) {
          logger.debug(`Array field without primary: ${prop}, value: ${propVal}`);
        } else {
          sanitizedDoc[`primaryOf${capitalizeFirstLetter(prop)}`] = primaryElement.sanitize
            ? primaryElement.sanitize()
            : primaryElement;
        }
      }
    } else {
      if (propVal != undefined) {
        sanitizedDoc[prop] = propVal.sanitize ? propVal.sanitize() : propVal;
      }
    }
  }
  return sanitizedDoc;
};

function isEmpty(field) {
  if (Array.isArray(field)) {
    if (field.length > 0) {
      const first = field[0];
      const concatString = Object.keys(first).map(key => (key == 'primary') ? '' : String(first[key])).join('');
      return (concatString == '');
    } else {
      return true;
    }
  } else return (field == '') || (field == null);
}

function removeEmpty(entityObj) {
  const nonEmptyKeys = Object.keys(entityObj).filter(key => !isEmpty(entityObj[key]));
  return _.pick(entityObj, nonEmptyKeys);
}

module.exports = {
  publishEntityUpdate,
  publishEntityView,
  removeEmpty,
  sanitizeDocument,
  initUpdateProducer,
  capitalizeFirstLetter,
};
