const addressFields = ['streetNumber', 'streetName', 'city', 'state', 'zipCode'];

function composeFullAddress(schema, options) {
  schema.pre('save', function (next) {
    if (addressFields.some(field => this.isModified(field))) {
      this.fullAddress = addressFields.map(field => (this[field] ? (' ' + this[field]) : '')).join('').trim(); // TODO better formatting for full address
    }
    return next();
  });
}

const nameFields = ['firstName', 'middleName', 'lastName'];

function composeFullName(schema, options) {
  schema.pre('save', function (next) {
    if (nameFields.some(field => this.isModified(field))) {
      this.fullName = nameFields.map(field => (this[field] ? (' ' + this[field]) : '')).join('').trim();
    }
    return next();
  });
}

function setType(schema, options) {
  schema.pre('save', function (next) {
    if (!this.type) {
      this.type = options.type;
    }
    return next();
  });
}

function setName(schema, options) {
  schema.pre('save', function (next) {
    if (this.isModified('names') || !this.name) {
      const name = this.names.find(n => n.primary);
      this.name = (name) ? name.fullName : '';
    }
    return next();
  });
}

function setSanitized(schema, options) {
  schema.pre('save', function (next) {
    this.sanitized = options.sanitizeFunction(this);
    return next();
  });
}

module.exports = {
  composeFullAddress,
  composeFullName,
  setName,
  setType,
  setSanitized,
};
