### Example containerisation of LSTM Covid Model API

This is a very simple implementation of a containerised model that takes in the scaled data, it uses python 3.11.9. 


#### Using the model

Make sure all of the necessary dependencies are installed: 
``` poetry install ```

Within the `src/lstm_covid_model/` folder, run 
``` uvicorn app:app --reload --host 0.0.0.0 --port 3000 ```

This will start the server. Now you can make a request

``` curl "http://localhost:3000/forecast?country=Chile&horizon=30" ```

and this should return:

```
{"country":"Chile","horizon":30,"start_date":"2020-08-30","dates":["2020-08-30","2020-08-31","2020-09-01","2020-09-02","2020-09-03","2020-09-04","2020-09-05","2020-09-06","2020-09-07","2020-09-08","2020-09-09","2020-09-10","2020-09-11","2020-09-12","2020-09-13","2020-09-14","2020-09-15","2020-09-16","2020-09-17","2020-09-18","2020-09-19","2020-09-20","2020-09-21","2020-09-22","2020-09-23","2020-09-24","2020-09-25","2020-09-26","2020-09-27","2020-09-28"],"predictions":[0,0,327,3086,118758,45334,25574,7856,46899,170583,8129,91134,290720,52751,0,10697,111778,95442,9546,46523,40784,68354,64007,155684,372054,76740,38066,27637,63679,59153]}
```

The "predictions" represent the output from the LSTM for the forecasted value per day of the 30-day interval. You can select another forecast horizon or another country by altering the API request above. 


### Using the model as a container

Within the `lstm-covid-model/` folder, run 
``` docker build -t lstm-model-api . 
    docker run --rm -p 3000:3000 lstm-model-api ```

Once you have portforwarded, you can use the request shown above to ensure that the API is working correctly. 