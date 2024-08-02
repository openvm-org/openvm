module.exports = {
    prettyPrintJson: function (jsonString) {
        try {
            const jsonObj = JSON.parse(jsonString);
            const sortedJsonObj = Object.keys(jsonObj).sort().reduce((acc, key) => {
                acc[key] = jsonObj[key];
                return acc;
            }, {});
            return JSON.stringify(sortedJsonObj, null, 2);
        } catch (error) {
            console.error("Invalid JSON string provided:", error);
            return jsonString;
        }
    }
};