const mongoose = require('mongoose');
const _ = require('lodash');
const {
  Name,
  SocialProfile,
  Connection,
} = require('./component-models');
const { setType, setName } = require('./plugins');

// add person-specific properties to base config to create the schema:
const personaSchema = new mongoose.Schema({
  names: [Name],
  socialProfiles: [SocialProfile],
  connections: [Connection],
}, {
  strict: false, // mongoose won't enforce the schema,
  timestamps: true //adds createdAt and updatedAt timestamps populates/updates them for you
});

personaSchema.plugin(setType, { type: 'persona' });
personaSchema.plugin(setName);

 const connectionSchema = new mongoose.Schema({
   fromEntityId: String,
   toEntityId: String,
 }, {
   strict: false,
   timestamps: true
 });

 connectionSchema.methods.sanitize = function () {
   return _.pick(this, ['fromEntityId', 'toEntityId']);
 };

module.exports = {
   connectionSchema,
   personaSchema,
 };
