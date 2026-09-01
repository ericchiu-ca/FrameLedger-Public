# 安全政策

FrameLedger 是尚未发布、本地优先的研究工具，处理可能敏感的源视频、帧、音频、
OCR、ASR 和研究证据。安全报告必须继续保护这些数据边界。

## 支持版本

| 版本 | 安全支持 |
|---|---|
| 首个 tag 之前的当前 `main` | 尽力修复 |
| Git tag 与 Release | 当前不存在 |

当前没有已发布稳定版本或安全维护周期。创建第一个 release tag 时必须更新本表。

## 报告漏洞

请勿为疑似漏洞创建公开 issue。请使用 GitHub Private Vulnerability Reporting：

https://github.com/ericchiu-ca/FrameLedger-Public/security/advisories/new

GitHub 只为 public repository 提供 Private Vulnerability Reporting。仓库可见性变为
public 后，所有者必须立即启用并测试此渠道；在确认前不得对外宣布仓库或接受外部贡献。

在条件允许时，请提供：

- 受影响的 commit、命令、平台和 Python 版本；
- 只使用合成或允许再分发数据的最小复现；
- 预期与实际发生的安全边界；
- 影响、前置条件，以及问题是否跨越信任边界；
- 建议修复或披露限制。

请勿附加会员视频、私人录制、真实 OCR/ASR 输出、模型权重、凭据、账户详情或专有研究。
请用净化后的 fixture 和 hash 替代。如果没有受限数据便无法复现，请先描述这个限制，
不要传输该数据。

当前不承诺固定响应或修复 SLA。维护者会先在 private advisory 中记录接收、验证状态、
范围和协调披露的下一步，再进行任何公开披露。

## 安全范围

能够合理破坏以下任一边界的问题属于报告范围：

- 源视频必须保持只读，不得被移动、重命名、覆盖、转码或在旁边写入 sidecar；
- 输出不得逃逸出显式授权的新输出目录、覆盖既有运行或修改 `freezes/phase1-v1`；
- path traversal、symlink、archive、文件名或 helper 输出处理不得导致任意文件读写；
- `workflow-ui` 必须绑定 `127.0.0.1`、使用会话认证、只允许一个 active batch、
  最多 50 个显式视频且严格串行；
- browser fallback import 和 batch ledger 必须留在会话选择的输出目标之下，
  且不得授权目录扫描；
- helper 协议、subprocess 参数、媒体 metadata、OCR/ASR 文本和 HTML 输出不得造成
  command、markup 或 script injection；
- 证据处理不得把源帧、音频、识别文本或研究产物发送到网络服务；
- offline model 选择、source/artifact SHA-256 绑定、coverage accounting 和
  fail-closed validation 不得被绕过；
- secret、credential、私有路径或受限证据不得进入 Git、构建产物、日志或 release asset；
- 畸形或恶意输入不得在已记录的文件、范围、批次和存储限制之外触发无界工作。

## 通常不在范围内

除非同时破坏上面的安全边界，下列情况通常不属于安全漏洞：

- OCR、ASR、routing、chapter 或金融语义准确性分歧；
- 非目标平台不受支持，或 Apple Vision/MLX runtime 不可用；
- 不伴随软件安全边界失效的法律、许可证、隐私或再分发争议；
- 没有证明在 FrameLedger 中可达影响的第三方 advisory；
- 显式授权且仍处于全部已记录限制内的有效任务所产生的资源成本。

不要使用无所有权或无授权的源媒体或账户进行测试。不要执行拒绝服务测试、公开
proof-of-concept 细节或访问他人数据。

## 披露与修复

确认有效的问题应在 private branch 或 private advisory fork 上修复并做回归测试。
Release note 必须描述已验证的影响和受影响版本，同时不暴露受限证据。只有完成验证后，
才考虑 CVE、patch release 或公开 advisory；本政策不预先承诺这些结果。

仓库当前没有生产服务或云部署。local loopback controller 不是生产 Web server；
Apple Vision、MLX、真实视频、file picker 和 source immutability 仍是发布时的人工验证门。
