// https://stackoverflow.com/questions/10865025/merge-flatten-an-array-of-arrays-in-javascript

function flatten(arr) {
  return arr.reduce(function (flat, toFlatten) {
    return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
  }, []);
}

module.exports = { flatten };