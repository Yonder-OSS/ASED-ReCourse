# PMP (Persona Management Platform)

## Responsibilities

Handles HTTP requests by providing CRUD operations to a mongo instance containing personas. 

## Goals

Create a service that is deployable to a Kubernetes cluster that interfaces with a DB store to manage personas.

## Setup

Install Docker and Docker-Compose

### Development

Install the node deps with:

`npm install`

You can run the app in watch mode with:

`sudo docker-compose -f docker-compose.yml build`

`sudo docker-compose -f docker-compose.yml up`

### Consumption

Run:

`sudo docker-compose build`

Docker-compose up the service with:

`sudo docker-compose up -d`

### Available CRUD Operations

    GET /personas
        returns all personas sorted by optional query param 'sort'

    GET /:id
        return persona of given type with given mongo id

    GET /:id/:element_id/weights
        return weights of persona with mongo id and socialProfile with element_id

    POST 
        Adds persona specified in req, returns object as serialized json 

    POST /:fromPersonaId/addConnection w/ body = {payload:{toPersonaId}}
        Adds connection from fromPersonaId to toPersonaId specified in body

    POST /:id/:field
        update a top-level field w/ request given mongo id

    POST /getBatchInfo
        POST a list of batch requests for persona info

    PUT /:id/:field
        update a top-level field w/ request given mongo id

    DELETE /:personaId 
        delete persona by id

    DELETE /:entityId/:field
        delete persona field

    DELETE /:entityId/:field/:elementId
        delete persona array field element

### Persona Data Model

    persona
        names: [Name],
        socialProfiles [SocialProfiles]

    Name
        primary: Boolean, // enforce only one primary
        firstName: String
        middleName: String
        lastName: String
        fullName: String

    SocialProfile
        name: String
        url: String
        type: String, i.e. Twitter, Facebook, Reddit, Instagram, Email, etc
        filename: String // contains the weights for this social profile
        collect: Boolean

