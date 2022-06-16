// copied from hermes

const personaReasons = [
  {
    display: 'Solicitation for Financial Assistance',
    value: 'Solicitation for Financial Assistance',
  },
  {
    display: 'Excessive Letter Writing - Postal',
    value: 'Excessive Letter Writing - Postal',
  },
  {
    display: 'Excessive Letter Writing - Email',
    value: 'Excessive Letter Writing - Email',
  },
  {
    display: 'Social Media Commentary',
    value: 'Social Media Commentary',
  },
  {
    display: 'Direct Phsyical Approach',
    value: 'Direct Phsyical Approach',
  },
  {
    display: 'Direct Physical Approach (Attempted)',
    value: 'Direct Physical Approach (Attempted)',
  },
  {
    display: 'Fixation / Unhealthy Interest in Client / Executives',
    value: 'Fixation / Unhealthy Interest in Client / Executives',
  },
  {
    display: 'Religious Ideologies',
    value: 'Religious Ideologies',
  },
  {
    display: 'Violent Threat (Direct or Indirect)',
    value: 'Violent Threat (Direct or Indirect)',
  },
  {
    display: 'Other',
    value: 'Other',
  },
];

const reasonsForInclusion = {
  persona: personaReasons,
};

const personaConnectionTypes = {
  persona: [
    {
      display: 'Former Colleague',
      value: 'Former Colleague',
    },
    {
      display: 'Attended Same University',
      value: 'Attended Same University',
    },
    {
      display: 'Lived in Same City',
      value: 'Lived in Same City',
    },
  ],
};


const connectionTypes = {
  persona: personaConnectionTypes,
};

const socialProfileTypes = [
  { display: 'Twitter', value: 'twitter' },
  { display: 'Facebook', value: 'facebook' },
  { display: 'Reddit', value: 'reddit' },
  { display: 'Instagram', value: 'instagram' },
  { display: 'Tumblr', value: 'tumblr' },
  { display: 'Other', value: 'other' },
];


module.exports = {
  connectionTypes,
  reasonsForInclusion,
  socialProfileTypes,
};