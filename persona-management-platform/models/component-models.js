const _ = require('lodash');
const mongoose = require('mongoose');
const {
  composeFullAddress,
  composeFullName,
  setSanitized,
} = require('./plugins');

const strict = (process.env.ENFORCE_COMPONENT_SCHEMA == 'true') || false;

exports.Name = new mongoose.Schema({
  sanitized: Object,
  primary: Boolean, // enforce only one primary
  firstName: { type: String, trim: true },
  middleName: { type: String, trim: true },
  lastName: { type: String, trim: true },
  fullName: String,
}, { strict });
exports.Name.plugin(composeFullName);

exports.Address = new mongoose.Schema({
  sanitized: Object,
  primary: Boolean,
  fromDate: Date,
  toDate: Date,
  streetNumber: Number,
  streetName: { type: String, trim: true },
  city: { type: String, trim: true },
  state: { type: String, trim: true },
  zipCode: Number,
  fullAddress: String, // set by plug-in
}, { strict });
exports.Address.plugin(composeFullAddress);
exports.Address.plugin(setSanitized, { sanitizeFunction: doc => doc.fullAddress });
exports.Address.methods.sanitize = function () {
  return this.fullAddress;
};

exports.Note = new mongoose.Schema({
  sanitized: Object,
  author: { type: String, trim: true },
  property: { type: String },
  propertyValue: { type: String },
  content: { type: String, trim: true },
}, { strict });
exports.Note.plugin(setSanitized, { sanitizeFunction: doc => _.pick(doc, ['property', 'propertyValue', 'content', 'belongsTo']) });
exports.Note.methods.sanitize = function () {
  return _.pick(this, ['property', 'propertyValue', 'content', 'belongsTo']);
};

exports.SocialProfile = new mongoose.Schema({
  sanitized: Object,
  name: { type: String, trim: true },
  url: { type: String, trim: true },
  filename: {type: String, trim: true},
  type: { type: String, trim: true },
  collect: { type: Boolean, default: true, index: true },
}, { strict });

exports.PhoneNumber = new mongoose.Schema({
  sanitized: Object,
  primary: Boolean,
  number: Number,
}, { strict });
exports.PhoneNumber.plugin(setSanitized, { sanitizeFunction: doc => String(doc.number) });
exports.PhoneNumber.methods.sanitize = function () {
  return String(this.number);
};

exports.Email = new mongoose.Schema({
  sanitized: Object,
  primary: Boolean,
  address: { type: String, trim: true },
}, { strict });
exports.Email.plugin(setSanitized, { sanitizeFunction: doc => doc.address });
exports.Email.methods.sanitize = function () {
  return this.address;
};

exports.Timespan = new mongoose.Schema({
  sanitized: Object,
  fromDate: Date,
  toDate: Date,
}, { strict });
exports.Timespan.plugin(setSanitized, { sanitizeFunction: doc => _.pick(doc, ['fromDate', 'toDate']) });
exports.Timespan.methods.sanitize = function () {
  return _.pick(this, ['fromDate', 'toDate']); // removed _id and any other extra props added by mongoose
};

exports.Connection = new mongoose.Schema({
  sanitized: Object,
  toEntityType: String,
  toEntityId: String,
  connectionType: { type: String, default: 'connected' },
}, { strict });
exports.Connection.plugin(setSanitized, { sanitizeFunction: doc => _.pick(doc, ['toEntityId']) });
exports.Connection.methods.sanitize = function () {
  return _.pick(this, ['toEntityId']);
};
