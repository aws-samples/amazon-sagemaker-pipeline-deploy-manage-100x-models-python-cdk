import os
import uuid
from helper import *
import json
import datetime
from datetime import timezone

def lambda_handler(event, context):
    print(event)
    try:
        inference_ddb_table = os.environ['inference_ddb_table']
        partition_key = os.environ['partition_key']

        body = event['Records'][0]['body']
        item_dict = json.loads(body)


    except Exception as e:
        print("We are using Dev environment: {}".format(e))
        print(event)
        body = event['Records'][0]['body']
        item_dict = json.loads(body)

        inference_ddb_table="InferenceResultsStack-inferencetable513B141C-1S8SO4SMYKJD3"
        partition_key = "id"

    desired_dim = item_dict.pop("desired_dim")
    ground_truth = item_dict.pop("Churn?_True.")

    csv_list = [value for key, value in item_dict.items()]
    payload= convert_list_to_csv_string(csv_list)

    desired_endpoint_name = get_desired_endpoint_name(desired_dim)

    query_response = query_endpoint(payload, desired_endpoint_name)
    prediction = parse_resonse(query_response)

    item_dict["prediction"] = round(prediction)
    item_dict["prediction_prob"] = prediction
    item_dict["ground_truth"] = ground_truth
    item_dict["desired_dim"] = desired_dim  
    
    # Get the current date and time
    current_datetime = datetime.datetime.now(timezone.utc)
    # Convert the datetime to a string format suitable for DynamoDB
    current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    item_dict["created_at"] = current_datetime_str

    item_dict[partition_key] = str(uuid.uuid4())
    print(item_dict)
    item_dict = convert_floats_to_decimals(item_dict)
    print(item_dict)

    write_single_item_to_dynamodb(item_dict, inference_ddb_table)  ## here we are overriding existing items  

    return {
        'statusCode': 200
    }

if __name__ == "__main__":
   event = {'Records': [{'messageId': '68c6ddd5-c087-4062-8fb7-259020dee36f', 'receiptHandle': 'AQEB8pXf/MbWQoeskFljRZbcagSw89HFNJOh/BLBhxAAivzg314KJHbewXQ8rSDlx1BBFkmmZyp+zqiQQ2JJ94Vij6r4Ej93x6juCvSKrMijG5G1LK9KJ68Tm2R33itoknjIcB/WRftShEwHq2Dd4QNcb1T+GQMYIJMCrhLnGXmtz9Lz61LhtLKkTAM1a23ueyNt2pkPYU7f2MTQ3EPIITjUL5uCkZLGv30P0JgdPmLmcn3ywRZ7jvq112hkViTsFaQowejnIS3F6Oa7l/rsZF4fSx2qDJ9A7kFLd5e/xQbkjIQKCuJ378GcjHxuc8qn2XL9HFbyEEtUjDMxncrVnFBmw3fD6GlBAN5pac4NWhJQTjg+Y4MJxmruS91t8bOIcLaTm3rHM3dx2t+Ao31HO5T8HuZbBkrbheOJbBnf3ChOtLZPNkJbsHgWo0l3Dk65ih5K', 'body': '{"Churn?_True.":0,"Account Length":62.0,"VMail Message":0.0,"Day Mins":5.07215206,"Day Calls":5.0,"Eve Mins":6.60041134,"Eve Calls":2.0,"Night Mins":3.53350108,"Night Calls":300.0,"Intl Mins":4.3952999,"Intl Calls":7.0,"CustServ Calls":6.0,"Dimension":1.0,"State_AK":0.0,"State_AL":0.0,"State_AR":0.0,"State_AZ":0.0,"State_CA":0.0,"State_CO":0.0,"State_CT":0.0,"State_DC":0.0,"State_DE":0.0,"State_FL":0.0,"State_GA":0.0,"State_HI":0.0,"State_IA":0.0,"State_ID":0.0,"State_IL":0.0,"State_IN":0.0,"State_KS":0.0,"State_KY":0.0,"State_LA":0.0,"State_MA":0.0,"State_MD":0.0,"State_ME":0.0,"State_MI":0.0,"State_MN":0.0,"State_MO":0.0,"State_MS":0.0,"State_MT":0.0,"State_NC":0.0,"State_ND":0.0,"State_NE":0.0,"State_NH":0.0,"State_NJ":0.0,"State_NM":0.0,"State_NV":0.0,"State_NY":0.0,"State_OH":0.0,"State_OK":0.0,"State_OR":0.0,"State_PA":0.0,"State_RI":0.0,"State_SC":0.0,"State_SD":0.0,"State_TN":0.0,"State_TX":0.0,"State_UT":0.0,"State_VA":0.0,"State_VT":1.0,"State_WA":0.0,"State_WI":0.0,"State_WV":0.0,"State_WY":0.0,"Area Code_657":0.0,"Area Code_658":0.0,"Area Code_659":0.0,"Area Code_676":0.0,"Area Code_677":0.0,"Area Code_678":0.0,"Area Code_686":0.0,"Area Code_707":0.0,"Area Code_716":0.0,"Area Code_727":0.0,"Area Code_736":0.0,"Area Code_737":0.0,"Area Code_758":0.0,"Area Code_766":0.0,"Area Code_776":0.0,"Area Code_777":0.0,"Area Code_778":0.0,"Area Code_786":0.0,"Area Code_787":0.0,"Area Code_788":0.0,"Area Code_797":0.0,"Area Code_798":0.0,"Area Code_806":0.0,"Area Code_827":0.0,"Area Code_836":0.0,"Area Code_847":0.0,"Area Code_848":0.0,"Area Code_858":0.0,"Area Code_866":0.0,"Area Code_868":1.0,"Area Code_876":0.0,"Area Code_877":0.0,"Area Code_878\'":0.0," \\"Int\'l Plan_no\\"":0.0," \\"Int\'l Plan_yes\\"":1.0," \'VMail Plan_no":1.0,"VMail Plan_yes":0.0,"desired_dim":"DummyDim2"}', 'attributes': {'ApproximateReceiveCount': '1', 'AWSTraceHeader': 'Root=1-658462ae-1d96b9c64c8467923aedc65c;Parent=40c9ce78273f8f4f;Sampled=0;Lineage=fec55efe:0', 'SentTimestamp': '1703174835373', 'SenderId': 'AROAS7TYIXZ52XDTNXXDG:InferenceResultsStack-SendMessagesToSQSD474CC7A-pqkKjReEOlXK', 'ApproximateFirstReceiveTimestamp': '1703174845373'}, 'messageAttributes': {}, 'md5OfBody': '69e8a76cb5228a1541b58c1f6bc611b8', 'eventSource': 'aws:sqs', 'eventSourceARN': 'arn:aws:sqs:eu-west-1:123456789012:InferenceResultsStack-RecordQueueCE027E6F-9rEpD04Gk0Ou', 'awsRegion': 'eu-west-1'}]}

   lambda_handler(event, None)