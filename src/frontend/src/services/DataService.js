import { BASE_API_URL } from "./Common";

const axios = require('axios');

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    GETSTARTUP: async function () {
        return await axios.get(BASE_API_URL + "/startup");
    },
    GetCurrentmodel: async function () {
        return await axios.get(BASE_API_URL + "/");
    },
    Predict: async function (file, textPrompt) {
        return await axios.post(BASE_API_URL + "/predict", file, textPrompt, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
}

export default DataService;