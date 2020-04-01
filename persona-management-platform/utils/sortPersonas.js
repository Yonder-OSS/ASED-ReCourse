const _ = require('lodash');

function sortPersonas(personas, sortBy = '-createdAt') {
  const reverse = (sortBy[0] == '-');
  const sortField = (reverse) ? sortBy.slice(1) : sortBy;
  sorted = _.sortBy(personas, sortPersona);

  if (reverse) sorted.reverse();
  return sorted;

  function sortPersona(persona) {
    if (sortField === 'name') {
      const name = persona.names.find(n => !!n.primary);
      return name ? name.fullName.toLowerCase() : '';
    } else {
      return persona[sortField];
    }
  }
}

module.exports = { sortPersonas };