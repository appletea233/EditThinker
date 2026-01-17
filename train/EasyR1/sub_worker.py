import ray
from ray.job_submission import JobSubmissionClient
import time
import argparse
import json
import sys
import re


def connect_to_ray(address=None):
    """连接到Ray集群"""
    try:
        if address:
            ray.init(address=address)
        else:
            ray.init()
        print("成功连接到Ray集群")
    except Exception as e:
        print(f"连接Ray集群失败: {e}")
        sys.exit(1)


def get_job_submission_client(address="http://127.0.0.1:8265"):
    """获取Job提交客户端"""
    try:
        client = JobSubmissionClient(address)
        return client
    except Exception as e:
        print(f"创建Job提交客户端失败: {e}")
        sys.exit(1)


def get_job_id(client):
    jobs = client.list_jobs()
    if not jobs:
        return None
    print(f"找到 {len(jobs)} 个作业:")
    for job in jobs:
        job_id = job.submission_id
        print(f" 提交ID: {job.submission_id}, 状态: {job.status}")
        if job_id is not None:
            return job_id
    return None


def get_job_info(client, job_id):
    """获取特定作业的详细信息"""
    try:
        info = client.get_job_info(job_id)
        print(f"\n作业 {job_id} 的详细信息:")
        print(f"  状态: {info.get('status', 'N/A')}")
        # print(f"  提交时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.get('start_time', 0)/1000))}")
        # if info.get('end_time'):
        #     print(f"  结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.get('end_time')/1000))}")
        # print(f"  入口点: {info.get('entrypoint', 'N/A')}")
        # print(f"  错误类型: {info.get('error_type', 'None')}")
        # print(f"  消息: {info.get('message', 'N/A')}")
        return info
    except Exception as e:
        print(f"获取作业信息失败: {e}")
        return None


def monitor_job(client, interval=30):
    job_id = None
    while job_id is None:
        job_id = get_job_id(client)

    """监控作业直到完成"""
    print(f"\n开始监控作业 {job_id}")
    try:
        while True:
            info = client.get_job_info(job_id)
            status = info.status
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 作业状态: {status}")

            if status in ["SUCCEEDED", "FAILED", "STOPPED", "CANCELED"]:
                print(f"作业已完成，最终状态: {status}")
                break

            time.sleep(interval)
        return status
    except KeyboardInterrupt:
        print("监控被用户中断")
        return None
    except Exception as e:
        print(f"监控作业失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Ray作业管理工具")
    parser.add_argument("--ray-address", help="Ray集群地址", default="127.0.0.1:8277")
    parser.add_argument("--job-server", help="作业服务器地址", default="http://127.0.0.1:8265")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    args = parser.parse_args()

    # 连接Ray集群
    connect_to_ray(args.ray_address)

    # 获取Job提交客户端
    client = get_job_submission_client(args.job_server)

    status = monitor_job(client)
    print(status)


if __name__ == "__main__":
    main()
