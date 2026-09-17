import json
import os

import boto3
from botocore.exceptions import ClientError


class Alerter:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled and os.getenv("ALERT_SNS_TOPIC_ARN")
        if self.enabled:
            self.sns = boto3.client("sns", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
            self.topic_arn = os.getenv("ALERT_SNS_TOPIC_ARN")

    def fire(self, reason: dict) -> str:
        msg = json.dumps(reason, default=lambda o: o.__dict__, indent=2)
        print(f"[ALERT] {msg}")
        if self.enabled:
            try:
                self.sns.publish(TopicArn=self.topic_arn, Message=msg, Subject="Vision Alert")
                return "sns_sent"
            except ClientError as e:
                print(f"[ERROR] SNS publish failed: {e}")
                return "sns_failed"
        return "local_only"
