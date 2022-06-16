const express = require('express');
const router = express.Router();

const personas = require('./personas');

router.get('/', function (req, res) {
  res.send('Persona Management Platform');
});

router.use('/', personas);

module.exports = router;
