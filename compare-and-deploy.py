import os
from model_deploy.deploy import deploy_model

MODEL_VERSION_URL = "/api/ml/api/2.0/mlflow/mlfoundry-artifacts/model-versions/list"
MODEL_ID = os.environ['MODEL_ID']
API_KEY = os.environ['TFY_API_KEY']
AUTH_SERVER_URL = "https://auth.production.truefoundry.com"
WORKSPACE_FQN = os.environ['WORKSPACE_FQN']

import mlfoundry as mlf
import requests
from urllib.parse import urljoin

client = mlf.get_client()

# get token from api key
token_response = requests.post(
    url=urljoin(AUTH_SERVER_URL, "/api/v1/oauth/api-keys/token"), json={"apiKey": API_KEY}
).json()
headers = {"Authorization": f"Bearer {token_response['accessToken']}"}


response_mv = requests.post(url=urljoin("https://app.truefoundry.com", MODEL_VERSION_URL), headers=headers, json={"model_id": MODEL_ID})
mv_info = {}
latest_fqn = None
mv_dict = response_mv.json()
for mv in mv_dict['model_versions']:
    metrics = client.get_run(mv['run_id']).get_metrics()
    mv_info[mv['fqn']] = {
        "accuracy": metrics['accuracy'][-1].value,
        "prediction_time": metrics['prediction_time'][-1].value,
    }
    latest_fqn = mv['fqn']


def get_best_mv(mv_info):
    best_fqn = None
    target_v = -10
    for fqn, mv in mv_info.items():
        current_v = mv['accuracy'] - mv['prediction_time']
        if current_v > target_v:
            best_fqn = fqn
            target_v = current_v
    return best_fqn


if get_best_mv(mv_info) == latest_fqn:
    deploy_model(workspace_fqn=WORKSPACE_FQN, model_version_fqn=latest_fqn)