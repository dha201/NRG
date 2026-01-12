import logging
from datetime import datetime, timezone

import azure.functions as func

app = func.FunctionApp()

def run_analysis():
    utc_timestamp = datetime.utcnow().replace(
        tzinfo=timezone.utc).isoformat()
    
    logging.info('NRG Legislative Analysis started at %s', utc_timestamp)
    
    try:
        # pylint: disable=import-outside-toplevel
        import legislative_tracker
        legislative_tracker.main()
        
        logging.info('NRG Legislative Analysis completed successfully')
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error('Error running legislative analysis: %s', str(e))
        raise

# pylint: disable=invalid-name
@app.route(route="run", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def NRGAnalysisHTTP(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger for on-demand testing
    Call via: https://<app-name>.azurewebsites.net/api/run?code=<function-key>
    """
    logging.info('HTTP trigger invoked by: %s', req.url)
    
    try:
        run_analysis()
        return func.HttpResponse(
            "NRG Legislative Analysis completed successfully",
            status_code=200
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return func.HttpResponse(
            f"Analysis failed: {str(e)}",
            status_code=500
        )

# pylint: disable=invalid-name
@app.schedule(schedule="0 0 8 * * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 
def NRGAnalysisTimer(myTimer: func.TimerRequest) -> None:
    """
    Timer trigger for scheduled daily execution
    Runs daily at 8:00 AM UTC
    CRON: 0 0 8 * * * (sec min hour day month dayOfWeek)
    """
    if myTimer.past_due:
        logging.info('The timer is past due!')
    
    logging.info('Timer trigger: Scheduled execution started')
    run_analysis()
